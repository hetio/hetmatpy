import numpy
import xarray

from .matrix import metaedge_to_adjacency_matrix


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
    source_node_ids, target_node_ids, adjacency_matrix = (
        metaedge_to_adjacency_matrix(graph, metaedge, dtype=dtype))

    dims = metaedge.source.identifier, metaedge.target.identifier
    coords = source_node_ids, target_node_ids

    data_array = xarray.DataArray(
        adjacency_matrix,
        coords=coords,
        dims=dims,
        name=metaedge.get_unicode_str()
    )
    return data_array
