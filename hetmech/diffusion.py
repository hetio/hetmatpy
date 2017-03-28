from collections import OrderedDict

import numpy
import hetio.hetnet


def dual_normalize(matrix,
                   row_damping=0,
                   column_damping=0,
                   copy=True):
    """
    Row and column normalize a 2d numpy array

    Parameters
    ==========
    matrix : numpy.array
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
    numpy.array
        Normalized matrix with dtype.float64.
    """
    # returns a newly allocated array
    matrix = matrix.astype(numpy.float64, copy=copy)

    # Normalize rows, unless row_damping is 0
    if row_damping != 0:
        row_sums = matrix.sum(axis=1)
        row_sums **= -row_damping
        # If row_sums contained zeros, now it contains Inf, so
        row_sums[numpy.isinf(row_sums)] = 0.0  # remove Inf
        # Reshape to normalize matrix by rows
        row_sums = row_sums.reshape((len(row_sums), 1))
        matrix *= row_sums

    # Normalize columns, unless column_damping is 0
    if column_damping != 0:
        column_sums = matrix.sum(axis=0)
        column_sums **= -column_damping
        # If column_sums contained zeros, now it contains Inf, so
        column_sums[numpy.isinf(column_sums)] = 0.0  # remove Inf
        # Reshape to normalize matrix by columns
        column_sums = column_sums.reshape((1, len(column_sums)))
        matrix *= column_sums

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


def diffuse_along_metapath(
        graph,
        metapath,
        source_node_weights,
        column_damping=1,
        row_damping=0,
        ):
    """
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
        adjacency_matrix = dual_normalize(
            adjacency_matrix, row_damping, column_damping)

        node_scores = adjacency_matrix @ node_scores

    target_metanode = metapath.target()
    target_node_to_position = get_node_to_position(graph, target_metanode)
    node_to_score = OrderedDict(zip(target_node_to_position, node_scores))
    return node_to_score
