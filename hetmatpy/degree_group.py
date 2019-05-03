import collections
import itertools

import numpy
import pandas
import scipy.sparse

from hetmatpy.matrix import metaedge_to_adjacency_matrix
import hetmatpy.degree_weight


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


def dwpc_to_degrees(graph, metapath, damping=0.5, ignore_zeros=False, ignore_redundant=True):
    """
    Yield a description of each cell in a DWPC matrix adding source and target
    node degree info as well as the corresponding path count.

    Parameters
    ----------
    ignore_redundant: bool
        When metapath is symmetric, only return a single orientation of a node pair.
        For example, yield source-target but not also target-source, which should have
        the same DWPC.
    """
    metapath = graph.metagraph.get_metapath(metapath)
    _, _, source_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[0], dense_threshold=0.7)
    _, _, target_adj_mat = metaedge_to_adjacency_matrix(graph, metapath[-1], dense_threshold=0.7)
    source_degrees = source_adj_mat.sum(axis=1).flat
    target_degrees = target_adj_mat.sum(axis=0).flat
    del source_adj_mat, target_adj_mat

    source_path = graph.get_nodes_path(metapath.source(), file_format='tsv')
    source_node_df = pandas.read_csv(source_path, sep='\t')
    source_node_names = list(source_node_df['name'])

    target_path = graph.get_nodes_path(metapath.target(), file_format='tsv')
    target_node_df = pandas.read_csv(target_path, sep='\t')
    target_node_names = list(target_node_df['name'])

    row_names, col_names, dwpc_matrix = graph.read_path_counts(metapath, 'dwpc', damping)
    dwpc_matrix = numpy.arcsinh(dwpc_matrix / dwpc_matrix.mean())
    if scipy.sparse.issparse(dwpc_matrix):
        dwpc_matrix = dwpc_matrix.toarray()

    _, _, path_count = graph.read_path_counts(metapath, 'dwpc', 0.0)
    if scipy.sparse.issparse(path_count):
        path_count = path_count.toarray()

    if ignore_redundant and metapath.is_symmetric():
        pairs = itertools.combinations_with_replacement(range(len(row_names)), 2)
    else:
        pairs = itertools.product(range(len(row_names)), range(len(col_names)))
    for row_ind, col_ind in pairs:
        dwpc_value = dwpc_matrix[row_ind, col_ind]
        if ignore_zeros and dwpc_value == 0:
            continue
        row = {
            'source_id': row_names[row_ind],
            'target_id': col_names[col_ind],
            'source_name': source_node_names[row_ind],
            'target_name': target_node_names[col_ind],
            'source_degree': source_degrees[row_ind],
            'target_degree': target_degrees[col_ind],
            'path_count': path_count[row_ind, col_ind],
            'dwpc': dwpc_value,
        }
        yield collections.OrderedDict(row)


def single_permutation_degree_group(permuted_hetmat, metapath, dwpc_mean, damping):
    """
    Compute degree-grouped permutations for a single permuted_hetmat,
    for one metapath.
    """
    _, _, matrix = hetmatpy.degree_weight.dwpc(permuted_hetmat, metapath, damping=damping, dense_threshold=0.7)
    source_deg_to_ind, target_deg_to_ind = hetmatpy.degree_group.metapath_to_degree_dicts(permuted_hetmat, metapath)
    row_generator = hetmatpy.degree_group.generate_degree_group_stats(
        source_deg_to_ind, target_deg_to_ind, matrix, scale=True, scaler=dwpc_mean)
    degree_grouped_df = (
        pandas.DataFrame(row_generator)
        .set_index(['source_degree', 'target_degree'])
        .assign(n_perms=1)
    )
    return degree_grouped_df
