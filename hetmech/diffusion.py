from collections import OrderedDict

import numpy
import hetio.hetnet


def diffusion_step(
        matrix, row_damping=0, column_damping=0, copy=True):
    """
    Return the diffusion adjacency matrix produced by the input matrix
    with the specified row and column normalization exponents.

    Parameters
    ==========
    matrix : numpy.ndarray
        adjacency matrix for a given metaedge, where the source nodes are
        columns and the target nodes are rows
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

    # Perform row normalization
    if row_damping != 0:
        row_sums = matrix.sum(axis=1)
        matrix = normalize(matrix, row_sums, 'rows', row_damping)

    # Perform column normalization
    if column_damping != 0:
        column_sums = matrix.sum(axis=0)
        matrix = normalize(matrix, column_sums, 'columns', column_damping)

    return matrix


def normalize(matrix, vector, axis, damping_exponent):
    """
    Normalize a 2D numpy.ndarray in place.

    Parameters
    ==========
    matrix : numpy.ndarray
    vector : numpy.ndarray
        Vector used for row or column normalization of matrix.
    axis : str
        'rows' or 'columns' for which axis to normalize
    """
    assert matrix.ndim == 2
    assert vector.ndim == 1
    if damping_exponent == 0:
        return matrix
    with numpy.errstate(divide='ignore'):
        vector **= -damping_exponent
    vector[numpy.isinf(vector)] = 0
    shape = (len(vector), 1) if axis == 'rows' else (1, len(vector))
    matrix *= vector.reshape(shape)
    return matrix


def get_node_to_position(graph, metanode):
    """
    Given a metanode, return a dictionary of node to position
    """
    if not isinstance(metanode, hetio.hetnet.MetaNode):
        # metanode is a name
        metanode = graph.node_dict(metanode)
    metanode_to_nodes = graph.get_metanode_to_nodes()
    nodes = sorted(metanode_to_nodes[metanode])
    node_to_position = OrderedDict((n, i) for i, n in enumerate(nodes))
    return node_to_position


def metaedge_to_adjacency_matrix(graph, metaedge, dtype=numpy.bool_):
    """
    Returns an adjacency matrix where source nodes are columns and target
    nodes are rows.
    """
    if not isinstance(metaedge, hetio.hetnet.MetaEdge):
        # metaedge is an abbreviation
        metaedge = graph.metagraph.metapath_from_abbrev(metaedge)[0]
    source_nodes = list(get_node_to_position(graph, metaedge.source))
    target_node_to_position = get_node_to_position(graph, metaedge.target)
    shape = len(target_node_to_position), len(source_nodes)
    adjacency_matrix = numpy.zeros(shape, dtype=dtype)
    for j, source_node in enumerate(source_nodes):
        for edge in source_node.edges[metaedge]:
            i = target_node_to_position[edge.target]
            adjacency_matrix[i, j] = 1
    return adjacency_matrix


def diffusion(
        graph,
        metapath,
        source_node_weights,
        column_damping=1,
        row_damping=0,
        ):
    """
    Performs diffusion from the specified source nodes.

    Parameters
    ==========
    graph : hetio.hetnet.Graph
        graph to extract adjacency matrixes along
    metapath : hetio.hetnet.MetaPath
        metapath to diffuse along
    source_node_weights : dict
        dictionary of node to weight. Nodes not in dict are zero-weighted
    column_damping : scalar
        exponent of (out)degree in column normalization
    row_damping : scalar
        exponent of (in)degree in row normalization
    """

    # Initialize node weights
    source_metanode = metapath.source()
    source_node_to_position = get_node_to_position(graph, source_metanode)
    node_scores = numpy.zeros(len(source_node_to_position))
    for source_node, weight in source_node_weights.items():
        i = source_node_to_position[source_node]
        node_scores[i] = weight

    for metaedge in metapath:
        adjacency_matrix = metaedge_to_adjacency_matrix(graph, metaedge)

        # Row/column normalization with degree damping
        adjacency_matrix = diffusion_step(
            adjacency_matrix, row_damping, column_damping)

        node_scores = adjacency_matrix @ node_scores

    target_metanode = metapath.target()
    target_node_to_position = get_node_to_position(graph, target_metanode)
    node_to_score = OrderedDict(zip(target_node_to_position, node_scores))
    return node_to_score
