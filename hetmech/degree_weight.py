import collections
import functools
import itertools
import operator

import numpy
from scipy import sparse

from .matrix import (normalize,
                     metaedge_to_adjacency_matrix,
                     auto_convert,
                     copy_array)


def dwwc_step(matrix, row_damping=0, column_damping=0, copy=True):
    """
    Return the degree-weighted adjacency matrix produced by the input matrix
    with the specified row and column normalization exponents.

    Parameters
    ==========
    matrix : numpy.ndarray or scipy.sparse
        adjacency matrix for a given metaedge, where the source nodes are
        rows and the target nodes are columns
    row_damping : int or float
        exponent to use in scaling each node's row by its in-degree
    column_damping : int or float
        exponent to use in scaling each node's column by its column-sum
    copy : bool
        `True` guarantees matrix will not be modified in place. `False`
        modifies in-place if and only if matrix.dtype == numpy.float64.
        Users are recommended not to rely on in-place conversion, but instead
        use `False` when in-place modification is acceptable and efficiency
        is desired.

    Returns
    =======
    numpy.ndarray or scipy.sparse
        Normalized matrix with dtype.float64.
    """
    # returns a newly allocated array
    matrix = copy_array(matrix, copy)

    row_sums = numpy.array(matrix.sum(axis=1)).flatten()
    column_sums = numpy.array(matrix.sum(axis=0)).flatten()
    matrix = normalize(matrix, row_sums, 'rows', row_damping)
    matrix = normalize(matrix, column_sums, 'columns', column_damping)

    return matrix


def dwwc(graph, metapath, damping=0.5, sparse_threshold=0):
    """
    Compute the degree-weighted walk count (DWWC). Special case of function
    dwpc_duplicated_metanode.
    """
    return dwpc_duplicated_metanode(graph, metapath, None, damping,
                                    sparse_threshold=sparse_threshold)


def pairwise(iterable):
    """
    Yield consecutive pairs of items from the iterable, but skip pairs where
    the items are equal.
    Modified from recipe in itertools docs at
    https://docs.python.org/3/library/itertools.html

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    for a, b in zip(a, b):
        if b != a:
            yield a, b


def get_segments(metagraph, metapath):
    """
    Break metapath into sub-metapaths that have at most one duplicate
    node type.
    """
    metanodes = metapath.get_nodes()
    metanode_to_indexes = collections.OrderedDict()
    for i, metanode in enumerate(metanodes):
        indexes = metanode_to_indexes.setdefault(metanode, [])
        indexes.append(i)

    # Ensure no overlapping metanode duplications
    last_stop = -1
    for metanode, indexes in metanode_to_indexes.items():
        if len(indexes) == 1:
            continue
        if min(indexes) <= last_stop:
            msg = ('Metapath f{metapath} contains overlapping'
                   'segments with duplicate metanodes.')
            raise ValueError(msg)
        last_stop = max(indexes)

    # Find indices to split at
    split_at = [0]
    range_to_duplicate = collections.OrderedDict()
    for metanode, indexes in metanode_to_indexes.items():
        if len(indexes) == 1:
            continue
        start = min(indexes)
        stop = max(indexes)
        range_to_duplicate[(start, stop)] = metanode
        split_at.append(start)
        split_at.append(stop)
    split_at.append(len(metapath))

    # Split at indices
    ranges = list(pairwise(split_at))
    segments = (metapath[start:stop] for start, stop in ranges)
    segments = [metagraph.get_metapath(metaedges) for metaedges in segments]
    duplicates = [range_to_duplicate.get(range_, None) for range_ in ranges]
    return segments, duplicates


def dwpc_duplicated_metanode(graph, metapath, duplicate=None, damping=0.5,
                             sparse_threshold=0):
    """
    Compute the degree-weighted path count (DWPC) when a single metanode is
    duplicated (any number of times). User must specify the duplicated
    metanode.

    Parameters
    ==========
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    duplicate : hetio.hetnet.MetaNode or None
    damping : float
    sparse_threshold : float (0 <= sparse_threshold <= 1)
        sets the density threshold above which a sparse matrix will be
        converted to a dense automatically.
    """
    if duplicate is not None:
        assert metapath.source() == duplicate
    dwpc_matrix = None
    row_names = None
    for metaedge in metapath:
        rows, cols, adj_mat = metaedge_to_adjacency_matrix(
            graph, metaedge, sparse_threshold=sparse_threshold)
        adj_mat = dwwc_step(adj_mat, damping, damping)
        if dwpc_matrix is None:
            row_names = rows
            dwpc_matrix = adj_mat
        else:
            dwpc_matrix = dwpc_matrix @ adj_mat
            dwpc_matrix = auto_convert(dwpc_matrix, sparse_threshold)
        if metaedge.target == duplicate:
            mat_type = type(dwpc_matrix)
            if mat_type == numpy.ndarray:
                mat_type = numpy.array
            diag_matrix = sparse.diags(dwpc_matrix.diagonal())
            dwpc_matrix = mat_type(dwpc_matrix - diag_matrix,
                                   dtype=numpy.float64, copy=False)

    dwpc_matrix = auto_convert(dwpc_matrix, sparse_threshold)
    return row_names, cols, dwpc_matrix


def dwpc(graph, metapath, damping=0.5, sparse_threshold=0):
    """
    Compute the degree-weighted path count (DWPC).
    """
    try:
        segments, duplicates = get_segments(graph.metagraph, metapath)
    except ValueError as e:
        raise NotImplementedError(e)

    parts = list()
    row_names = None
    for segment, duplicate in zip(segments, duplicates):
        if duplicate is None:
            rows, cols, matrix = dwwc(
                graph, segment, damping, sparse_threshold=sparse_threshold)
        else:
            rows, cols, matrix = dwpc_duplicated_metanode(
                graph, segment, duplicate, damping,
                sparse_threshold=sparse_threshold)
        if row_names is None:
            row_names = rows
        parts.append(matrix)

    dwpc_matrix = functools.reduce(operator.matmul, parts)
    return row_names, cols, dwpc_matrix
