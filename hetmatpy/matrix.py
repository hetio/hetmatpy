import hetnetpy.hetnet
import hetnetpy.matrix
import hetnetpy.permute
import numpy
import scipy.sparse

import hetmatpy.hetmat


def metaedge_to_adjacency_matrix(graph_or_hetmat, *args, **kwargs):
    """
    Return an adjacency matrix tuple like (rows, cols, matrix) for a specified
    metapath. This function is a compatibility wrapper allowing
    graph_or_hetmat to be either a hetnetpy.hetnet.Graph or hetmatpy.hetmat.HetMat.
    """
    if isinstance(graph_or_hetmat, hetmatpy.hetmat.HetMat):
        return graph_or_hetmat.metaedge_to_adjacency_matrix(*args, **kwargs)
    if isinstance(graph_or_hetmat, hetnetpy.hetnet.Graph):
        return hetnetpy.matrix.metaedge_to_adjacency_matrix(
            graph_or_hetmat, *args, **kwargs
        )
    raise TypeError(f"graph_or_hetmat is an unsupported type: {type(graph_or_hetmat)}")


def get_node_identifiers(graph_or_hetmat, metanode):
    """
    Return node identifiers for a given metanode.
    """
    metanode = graph_or_hetmat.metagraph.get_metanode(metanode)
    if isinstance(graph_or_hetmat, hetmatpy.hetmat.HetMat):
        return graph_or_hetmat.get_node_identifiers(metanode)
    if isinstance(graph_or_hetmat, hetnetpy.hetnet.Graph):
        return hetnetpy.matrix.get_node_identifiers(graph_or_hetmat, metanode)
    raise TypeError(f"graph_or_hetmat is an unsupported type: {type(graph_or_hetmat)}")


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
    with numpy.errstate(divide="ignore"):
        vector **= -damping_exponent
    vector[numpy.isinf(vector)] = 0
    vector = scipy.sparse.diags(vector)
    if axis == "rows":
        # equivalent to `vector @ matrix` but returns scipy.sparse.csc not scipy.sparse.csr  # noqa: E501
        matrix = (matrix.transpose() @ vector).transpose()
    else:
        matrix = matrix @ vector
    return matrix


def copy_array(matrix, copy=True, dtype=numpy.float64):
    """Returns a newly allocated array if copy is True"""
    assert matrix.ndim == 2
    assert matrix.dtype != "O"  # Ensures no empty row
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


def permute_matrix(
    adjacency_matrix, directed=False, multiplier=10, excluded_pair_set=set(), seed=0
):
    """
    Perform a degree-preserving permutation on a given adjacency matrix. Assumes
    boolean matrix, and is incompatible with weighted edges.

    Parameters
    ----------
    adjacency_matrix : numpy.ndarray or scipy.sparse
    directed : bool
    multiplier : float
        Number of times to try edge swaps as a fraction of the number of edges.
        Default is ten times the number of tries as edges.
    excluded_pair_set : set
        Pairs of nodes to exclude from the permutation
    seed : int

    Returns
    -------
    numpy.ndarray or scipy.sparse, list
        Permuted adjacency matrix of the same type as was passed. List of
        OrderedDicts of information on the permutations performed.
    """
    edge_list = list(zip(*adjacency_matrix.nonzero()))
    permuted_edges, stats = hetnetpy.permute.permute_pair_list(
        edge_list,
        directed=directed,
        multiplier=multiplier,
        excluded_pair_set=excluded_pair_set,
        seed=seed,
    )

    edges = numpy.array(permuted_edges)
    ones = numpy.ones(len(edges), dtype=adjacency_matrix.dtype)
    permuted_adjacency = scipy.sparse.csc_matrix(
        (ones, (edges[:, 0], edges[:, 1])), shape=adjacency_matrix.shape
    )

    # Keep the same sparse type as adjacency_matrix
    if scipy.sparse.issparse(adjacency_matrix):
        permuted_adjacency = type(adjacency_matrix)(permuted_adjacency)
    else:
        permuted_adjacency = permuted_adjacency.toarray()

    # Ensure node degrees have been preserved
    assert (permuted_adjacency.sum(axis=1) == adjacency_matrix.sum(axis=1)).all()
    assert (permuted_adjacency.sum(axis=0) == adjacency_matrix.sum(axis=0)).all()

    return permuted_adjacency, stats
