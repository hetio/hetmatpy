import numpy

from .matrix import normalize, metaedge_to_adjacency_matrix


def dwwc_step(
        matrix, row_damping=0, column_damping=0, copy=True):
    """
    Return the degree-weighted adjacency matrix produced by the input matrix
    with the specified row and column normalization exponents.

    Parameters
    ==========
    matrix : numpy.ndarray
        adjacency matrix for a given metaedge, where the source nodes are
        rows and the target nodes are columns
    row_damping : int or float
        exponent to use in scaling each node's row by its in-degree
    column_damping : int or float
        exponent to use in scaling each node's column by its column-sum
    copy : bool
        `True` gaurantees matrix will not be modified in place. `False`
        modifies in-place if and only if matrix.dtype == numpy.float64.
        Users are recommended not to rely on in-place conversion, but instead
        use `False` when in-place modification is acceptable and efficiency
        is desired.

    Returns
    =======
    numpy.ndarray
        Normalized matrix with dtype.float64.
    """
    # returns a newly allocated numpy.ndarray
    matrix = numpy.array(matrix, numpy.float64, copy=copy)
    assert matrix.ndim == 2

    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)
    matrix = normalize(matrix, row_sums, 'rows', row_damping)
    matrix = normalize(matrix, column_sums, 'columns', column_damping)

    return matrix


def dwwc(graph, metapath, damping=0.5):
    """
    Compute the degree-weighted walk count (DWWC).
    """
    dwwc_matrix = None
    row_names = None
    for metaedge in metapath:
        rows, cols, adj_mat = metaedge_to_adjacency_matrix(graph, metaedge)
        adj_mat = dwwc_step(adj_mat, damping, damping)
        if dwwc_matrix is None:
            row_names = rows
            dwwc_matrix = adj_mat
        else:
            dwwc_matrix = dwwc_matrix @ adj_mat
    return row_names, cols, dwwc_matrix
