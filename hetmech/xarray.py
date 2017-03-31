import numpy
import xarray
import hetio.hetnet

from .diffusion import get_node_to_position


def graph_to_xarray(graph):
    """
    Convert a hetio.hetnet.Graph to an xarray.Dataset
    """
    data_vars = dict()
    for metaedge in graph.metagraph.get_edges(exclude_inverts=True):
        data_array = metaedge_to_data_array(graph, metaedge)
        name = metaedge.get_abbrev()
        data_vars[name] = data_array
    dataset = xarray.Dataset(data_vars)
    return dataset


def metaedge_to_data_array(graph, metaedge, dtype=numpy.bool_):
    """
    Return an xarray.DataArray that's an adjacency matrix where source nodes
    are columns and target nodes are rows.
    """
    if not isinstance(metaedge, hetio.hetnet.MetaEdge):
        # metaedge is an abbreviation
        metaedge = graph.metagraph.metapath_from_abbrev(metaedge)[0]

    source_nodes = list(get_node_to_position(graph, metaedge.source))
    target_node_to_position = get_node_to_position(graph, metaedge.target)
    shape = len(source_nodes), len(target_node_to_position)
    adjacency_matrix = numpy.zeros(shape, dtype=dtype)
    for i, source_node in enumerate(source_nodes):
        for edge in source_node.edges[metaedge]:
            j = target_node_to_position[edge.target]
            adjacency_matrix[i, j] = 1

    dims = metaedge.source.identifier, metaedge.target.identifier
    coords = (
        [node.identifier for node in source_nodes],
        [node.identifier for node in target_node_to_position],
    )

    data_array = xarray.DataArray(
        adjacency_matrix,
        coords=coords,
        dims=dims,
        name=metaedge.get_unicode_str()
    )
    return data_array
