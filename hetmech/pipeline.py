import pandas
import scipy.special
import scipy.stats

import hetmech.degree_group
import hetmech.degree_weight
import hetmech.hetmat


def combine_dwpc_dgp(graph, metapath, damping, ignore_zeros=False, max_p_value=1.0):
    """
    Combine DWPC information with degree-grouped permutation summary metrics.
    Includes gamma-hurdle significance estimates.
    """
    stats_path = graph.get_running_degree_group_path(metapath, 'dwpc', damping, extension='.tsv.gz')
    dgp_df = pandas.read_table(stats_path)
    dgp_df['mean_nz'] = dgp_df['sum'] / dgp_df['nnz']
    dgp_df['sd_nz'] = ((dgp_df['sum_of_squares'] - dgp_df['sum'] ** 2 / dgp_df['nnz']) / (dgp_df['nnz'] - 1)) ** 0.5
    dgp_df['beta'] = dgp_df['mean_nz'] / dgp_df['sd_nz'] ** 2
    dgp_df['alpha'] = dgp_df['mean_nz'] * dgp_df['beta']
    degrees_to_dgp = dgp_df.set_index(['source_degree', 'target_degree']).to_dict(orient='index')
    dwpc_row_generator = hetmech.degree_group.dwpc_to_degrees(
        graph, metapath, damping=damping, ignore_zeros=ignore_zeros)
    for row in dwpc_row_generator:
        degrees = row['source_degree'], row['target_degree']
        dgp = degrees_to_dgp[degrees]
        row.update(dgp)
        if row['path_count'] == 0:
            row['p_value'] = 1.0
        else:
            row['p_value'] = None if row['sum'] == 0 else (
                row['nnz'] / row['n'] *
                (1 - scipy.special.gammainc(row['alpha'], row['beta'] * row['dwpc']))
            )
        if row['p_value'] is not None and row['p_value'] > max_p_value:
            continue
        for key in ['sum', 'sum_of_squares', 'beta', 'alpha']:
            del row[key]
        yield row
