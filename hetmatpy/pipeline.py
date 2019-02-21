import itertools

import pandas
import scipy.special
import scipy.stats

import hetmatpy.degree_group
import hetmatpy.degree_weight
import hetmatpy.hetmat

FLOAT_ERROR_TOLERANCE = 1e-5


def calculate_sd(sum_of_squares, unsquared_sum, number_nonzero):
    """
    Calculate the standard deviation and validate the incoming data
    """
    if number_nonzero == 1:
        return None

    squared_deviations = sum_of_squares - unsquared_sum ** 2 / number_nonzero

    # If all the values in the row are the same we'll manually return zero,
    # because not doing so can lead to some issues with float imprecision
    # The true value of the squared deviation will always be >= zero,
    # but float error may bring it below zero
    if abs(squared_deviations) < FLOAT_ERROR_TOLERANCE:
        return 0
    else:
        return (squared_deviations / (number_nonzero - 1)) ** 0.5


def add_gamma_hurdle_to_dgp_df(dgp_df):
    """
    Edit a degree-grouped permutation dataframe to include gamma-hurdle
    distribution parameters.
    """
    # Validate dgp_df
    if not isinstance(dgp_df, pandas.DataFrame):
        raise ValueError('add_gamma_hurdle_to_dgp_df: dgp_df must be a pandas.DataFrame')
    missing = {'nnz', 'sum', 'sum_of_squares'} - set(dgp_df.columns)
    if missing:
        raise ValueError(
            'add_gamma_hurdle_to_dgp_df: '
            'dgp_df missing the following required columns: ' +
            ', '.join(missing)
        )
    # Compute gamma-hurdle parameters
    dgp_df['mean_nz'] = dgp_df['sum'] / dgp_df['nnz']
    dgp_df['sd_nz'] = calculate_sd(dgp_df['sum_of_squares'], dgp_df['sum'], dgp_df['nnz'])

    # If the standard deviation is zero, we'll go ahead and set beta and alpha to None
    # so that the gamma function breaks if it is called
    if dgp_df['sd_nz'] == 0 or dgp_df['sd_nz'] is None:
        dgp_df['beta'] = None
        dgp_df['alpha'] = None
    else:
        dgp_df['beta'] = dgp_df['mean_nz'] / dgp_df['sd_nz'] ** 2
        dgp_df['alpha'] = dgp_df['mean_nz'] * dgp_df['beta']
    return dgp_df


def calculate_gamma_hurdle_p_value(row):
    """
    Use the gamma hurdle model to calculate the p_value for a metapath
    """
    return row['nnz'] / row['n'] * (scipy.special.gammaincc(row['alpha'], row['beta'] * row['dwpc']))


def calculate_empirical_p_value(row):
    """
    Calculate p_value in cases where the gamma hurdle model won't work
    """
    if row['path_count'] == 0:
        # No paths exist between the given source and target nodes
        return 1.0
    if row['nnz'] == 0:
        # No nonzero DWPCs are found in the permuted network, but paths are
        # observed in the true network
        return 0.0
    if row['sd_nz'] == 0:
        # The DWPCs in the permuted network are identical
        if row['dwpc'] <= row['mean_nz']:
            # The DWPC you found in the true network is smaller than or equal
            # to those in the permuted network
            return row['nnz'] / row['n_dwpcs']

        # The DWPC you found in the true network is larger than those in the
        # permuted network
        return 0.0
    raise NotImplementedError


def calculate_p_value(row):
    """
    Calculate the p_value for a given metapath
    """
    if row['nnz'] == 0 or row['path_count'] == 0 or row['sd_nz'] == 0:
        return calculate_empirical_p_value(row)
    else:
        return calculate_gamma_hurdle_p_value(row)


def combine_dwpc_dgp(graph, metapath, damping, ignore_zeros=False, max_p_value=1.0):
    """
    Combine DWPC information with degree-grouped permutation summary metrics.
    Includes gamma-hurdle significance estimates.
    """
    stats_path = graph.get_running_degree_group_path(metapath, 'dwpc', damping, extension='.tsv.gz')
    dgp_df = pandas.read_csv(stats_path, sep='\t')
    dgp_df = add_gamma_hurdle_to_dgp_df(dgp_df)
    degrees_to_dgp = dgp_df.set_index(['source_degree', 'target_degree']).to_dict(orient='index')
    dwpc_row_generator = hetmatpy.degree_group.dwpc_to_degrees(
        graph, metapath, damping=damping, ignore_zeros=ignore_zeros)
    for row in dwpc_row_generator:
        degrees = row['source_degree'], row['target_degree']
        dgp = degrees_to_dgp[degrees]
        row.update(dgp)
        row['p_value'] = calculate_p_value(row)
        if row['p_value'] is not None and row['p_value'] > max_p_value:
            continue
        for key in ['sum', 'sum_of_squares', 'beta', 'alpha']:
            del row[key]
        yield row


def grouper(iterable, group_size):
    """
    Group an iterable into chunks of group_size.
    Derived from https://stackoverflow.com/a/8998040/4651668
    """
    iterable = iter(iterable)
    while True:
        chunk = itertools.islice(iterable, group_size)
        try:
            head = next(chunk),
        except StopIteration:
            break
        yield itertools.chain(head, chunk)


def grouped_tsv_writer(row_generator, path, group_size=20_000, sep='\t', index=False, **kwargs):
    """
    Write an iterable of dictionaries to a TSV, where each dictionary is a row.
    Uses pandas (extra keyword arguments are passed to DataFrame.to_csv) to
    write the TSV, enabling using pandas to write a generated rows that are too
    plentiful to fit in memory.
    """
    chunks = grouper(row_generator, group_size=group_size)
    for i, chunk in enumerate(chunks):
        df = pandas.DataFrame.from_records(chunk)
        kwargs['header'] = not bool(i)
        kwargs['mode'] = 'a' if i else 'w'
        df.to_csv(path, sep=sep, index=index, **kwargs)
