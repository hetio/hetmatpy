import numpy
import scipy.sparse

import hetio.hetnet
import hetio.matrix

import hetmech.hetmat


def metaedge_to_adjacency_matrix(graph_or_hetmat, *args, **kwargs):
    """
    Return an adjacency matrix tuple like (rows, cols, matrix) for a specified
    metapath. This function is a compatibility wrapper allowing
    graph_or_hetmat to be either a hetio.hetnet.Graph or hetmech.hetmat.HetMat.
    """
    if isinstance(graph_or_hetmat, hetmech.hetmat.HetMat):
        return graph_or_hetmat.metaedge_to_adjacency_matrix(*args, **kwargs)
    if isinstance(graph_or_hetmat, hetio.hetnet.Graph):
        return hetio.matrix.metaedge_to_adjacency_matrix(graph_or_hetmat, *args, **kwargs)
    raise TypeError(f'graph_or_hetmat is an unsupported type: {type(graph_or_hetmat)}')


def normalize(matrix, vector, axis, damping_exponent):
    """
    Normalize a 2D numpy.ndarray.

    Parameters
    ==========
    matrix : numpy.ndarray or scipy.sparse
    vector : numpy.ndarray
        Vector used for row or column normalization of matrix.
    axis : str
        'rows' or 'columns' for which axis to normalize
    damping_exponent : float
        exponent to use in scaling a node's row or column
    """
    assert matrix.ndim == 2
    assert vector.ndim == 1
    if damping_exponent == 0:
        return matrix
    with numpy.errstate(divide='ignore'):
        vector **= -damping_exponent
    vector[numpy.isinf(vector)] = 0
    vector = scipy.sparse.diags(vector)
    if axis == 'rows':
        # equivalent to `vector @ matrix` but returns scipy.sparse.csc not scipy.sparse.csr  # noqa: E501
        matrix = (matrix.transpose() @ vector).transpose()
    else:
        matrix = matrix @ vector
    return matrix


def copy_array(matrix, copy=True, dtype=numpy.float64):
    """Returns a newly allocated array if copy is True"""
    assert matrix.ndim == 2
    assert matrix.dtype != 'O'  # Ensures no empty row
    if not scipy.sparse.issparse(matrix):
        assert numpy.isfinite(matrix).all()  # Checks NaN and Inf
    try:
        matrix[0, 0]  # Checks that there is a value in the matrix
    except IndexError:
        raise AssertionError("Array may have empty rows")

    mat_type = type(matrix)
    if mat_type == numpy.ndarray:
        mat_type = numpy.array
    matrix = mat_type(matrix, dtype=dtype, copy=copy)
    return matrix
