from collections import OrderedDict

import hetio.hetnet
import numpy
from scipy import sparse


def get_node_to_position(graph, metanode):
    """
    Given a metanode, return a dictionary of node to position
    """
    if not isinstance(metanode, hetio.hetnet.MetaNode):
        # metanode is a name
        metanode = graph.metagraph.node_dict[metanode]
    metanode_to_nodes = graph.get_metanode_to_nodes()
    nodes = sorted(metanode_to_nodes[metanode])
    node_to_position = OrderedDict((n, i) for i, n in enumerate(nodes))
    return node_to_position


def metaedge_to_adjacency_matrix(graph, metaedge, dtype=numpy.bool_,
                                 matrix_type=numpy.array):
    """
    Returns an adjacency matrix where source nodes are rows and target
    nodes are columns.
    """
    if not isinstance(metaedge, hetio.hetnet.MetaEdge):
        # metaedge is an abbreviation
        metaedge = graph.metagraph.metapath_from_abbrev(metaedge)[0]
    source_nodes = list(get_node_to_position(graph, metaedge.source))
    target_node_to_position = get_node_to_position(graph, metaedge.target)
    shape = len(source_nodes), len(target_node_to_position)
    row, col, data = [], [], []
    for i, source_node in enumerate(source_nodes):
        for edge in source_node.edges[metaedge]:
            row.append(i)
            col.append(target_node_to_position[edge.target])
            data.append(1)
    adjacency_matrix = sparse.csc_matrix((data, (row, col)), shape=shape,
                                         dtype=dtype)
    if matrix_type == numpy.array or matrix_type == numpy.ndarray:
        adjacency_matrix = adjacency_matrix.toarray()
    elif matrix_type == numpy.matrix:
        adjacency_matrix = adjacency_matrix.todense()
    else:
        adjacency_matrix = matrix_type(adjacency_matrix)
    row_names = [node.identifier for node in source_nodes]
    column_names = [node.identifier for node in target_node_to_position]
    return row_names, column_names, adjacency_matrix


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
