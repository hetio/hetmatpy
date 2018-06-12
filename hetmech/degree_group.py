import itertools

import numpy
import pandas
import scipy.sparse

from hetmech.matrix import metaedge_to_adjacency_matrix
import hetmech.degree_weight


def degrees_to_degree_to_ind(degrees):
    degree_to_indices = dict()
    for i, degree in sorted(enumerate(degrees), key=lambda x: x[1]):
        degree_to_indices.setdefault(degree, []).append(i)
    return degree_to_indices


def metapath_to_degree_dicts(graph, metapath):
    metapath = graph.metagraph.get_metapath(metapath)
    _, _, source_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[0], dense_threshold=0.7)
    _, _, target_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[-1], dense_threshold=0.7)
    source_degrees = source_adj_mat.sum(axis=1).flat
    target_degrees = target_adj_mat.sum(axis=0).flat
    source_degree_to_ind = degrees_to_degree_to_ind(source_degrees)
    target_degree_to_ind = degrees_to_degree_to_ind(target_degrees)
    return source_degree_to_ind, target_degree_to_ind


def generate_degree_group_stats(source_degree_to_ind, target_degree_to_ind, matrix, scale=False, scaler=1):
    """
    Yield dictionaries with degree grouped stats
    """
    if scipy.sparse.issparse(matrix) and not scipy.sparse.isspmatrix_csr(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)
    for source_degree, row_inds in source_degree_to_ind.items():
        if source_degree > 0:
            row_matrix = matrix[row_inds, :]
            if scipy.sparse.issparse(row_matrix):
                row_matrix = row_matrix.toarray()
                # row_matrix = scipy.sparse.csc_matrix(row_matrix)
        for target_degree, col_inds in target_degree_to_ind.items():
            row = {
                'source_degree': source_degree,
                'target_degree': target_degree,
            }
            row['n'] = len(row_inds) * len(col_inds)
            if source_degree == 0 or target_degree == 0:
                row['sum'] = 0
                row['nnz'] = 0
                row['sum_of_squares'] = 0
                yield row
                continue

            slice_matrix = row_matrix[:, col_inds]
            values = slice_matrix.data if scipy.sparse.issparse(slice_matrix) else slice_matrix
            if scale:
                values = numpy.arcsinh(values / scaler)
            row['sum'] = values.sum()
            row['sum_of_squares'] = (values ** 2).sum()
            if scipy.sparse.issparse(slice_matrix):
                row['nnz'] = slice_matrix.nnz
            else:
                row['nnz'] = numpy.count_nonzero(slice_matrix)
            yield row


def compute_summary_metrics(df):
    df['mean'] = df['sum'] / df['n']
    df['sd'] = ((df['sum_of_squares'] - df['sum'] ** 2 / df['n']) / (df['n'] - 1)) ** 0.5
    return df


def dwpc_to_degrees(graph, metapath, damping=0.5):
    metapath = graph.metagraph.get_metapath(metapath)

    row_names, col_names, dwpc_matrix = graph.read_path_counts(metapath, 'dwpc', damping)
    _, _, path_count = graph.read_path_counts(metapath, 'dwpc', 0.0)
    _, _, source_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[0], dense_threshold=0.7)
    _, _, target_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[-1], dense_threshold=0.7)
    source_degrees = source_adj_mat.sum(axis=1).flat
    target_degrees = target_adj_mat.sum(axis=0).flat

    source_path = graph.get_nodes_path(metapath.source(), file_format='tsv')
    source_node_df = pandas.read_table(source_path)
    source_node_names = list(source_node_df['name'])

    target_path = graph.get_nodes_path(metapath.target(), file_format='tsv')
    target_node_df = pandas.read_table(target_path)
    target_node_names = list(target_node_df['name'])

    dwpc_matrix = numpy.arcsinh(dwpc_matrix / dwpc_matrix.mean())
    if scipy.sparse.issparse(dwpc_matrix):
        dwpc_matrix = dwpc_matrix.toarray()
    if scipy.sparse.issparse(path_count):
        path_count = path_count.toarray()
    row_inds, col_inds = range(len(row_names)), range(len(col_names))
    for row in itertools.product(row_inds, col_inds):
        row_ind, col_ind = row
        row = {
            'source_id': row_names[row_ind],
            'source_name': source_node_names[row_ind],
            'target_name': target_node_names[col_ind],
            'target_id': col_names[col_ind],
            'source_degree': source_degrees[row_ind],
            'target_degree': target_degrees[col_ind],
            'dwpc': dwpc_matrix[row_ind, col_ind],
            'path-count': path_count[row_ind, col_ind],
        }
        yield row
        continue


def single_permutation_degree_group(permuted_hetmat, metapath, dwpc_mean, damping):
    """
    Compute degree-grouped permutations for a single permuted_hetmat,
    for one metapath.
    """
    _, _, matrix = hetmech.degree_weight.dwpc(permuted_hetmat, metapath, damping=damping, dense_threshold=0.7)
    source_deg_to_ind, target_deg_to_ind = hetmech.degree_group.metapath_to_degree_dicts(permuted_hetmat, metapath)
    row_generator = hetmech.degree_group.generate_degree_group_stats(
        source_deg_to_ind, target_deg_to_ind, matrix, scale=True, scaler=dwpc_mean)
    degree_grouped_df = (
        pandas.DataFrame(row_generator)
        .set_index(['source_degree', 'target_degree'])
    )
    return degree_grouped_df


def summarize_degree_grouped_permutations(graph, metapath, damping):
    """
    Combine degree-grouped permutation information from all permutations into
    a single file per metapath.
    """
    degree_stats_df = None
    for permat in graph.permutations.values():
        path = permat.directory.joinpath('degree-grouped-path-counts', f'dwpc-{float(damping)}',
                                         f'{metapath}.tsv')
        df = (
            pandas.read_table(path)
            .set_index(['source_degree', 'target_degree'])
            .assign(n_perms=1)
        )
        if degree_stats_df is None:
            degree_stats_df = df
        else:
            degree_stats_df += df
    degree_stats_df = hetmech.degree_group.compute_summary_metrics(degree_stats_df)
    degree_stats_df.drop(columns=['sum', 'sum_of_squares'], inplace=True)
    return degree_stats_df
