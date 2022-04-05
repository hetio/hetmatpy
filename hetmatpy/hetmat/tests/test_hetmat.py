import numpy

import hetmatpy.hetmat
import hetmatpy.matrix
from hetmatpy.testing import get_graph


def test_disease_gene_example_conversion_to_hetmat(tmpdir):
    """
    Test converting the hetmat from Figure 2C of https://doi.org/crz8 into a
    hetmat.
    """
    graph = get_graph("disease-gene-example")
    hetmat = hetmatpy.hetmat.hetmat_from_graph(graph, tmpdir)
    assert list(graph.metagraph.get_nodes()) == list(hetmat.metagraph.get_nodes())

    # Test GaD adjacency matrix
    hetnet_adj = hetmatpy.matrix.metaedge_to_adjacency_matrix(
        graph, "GaD", dense_threshold=0
    )
    hetmat_adj = hetmatpy.matrix.metaedge_to_adjacency_matrix(
        hetmat, "GaD", dense_threshold=0
    )
    assert hetnet_adj[0] == hetmat_adj[0]  # row identifiers
    assert hetnet_adj[1] == hetmat_adj[1]  # column identifiers
    assert numpy.array_equal(hetnet_adj[2], hetmat_adj[2])  # adj matrices

    # Test DaG adjacency matrix (hetmat only stores GaD and must transpose)
    hetnet_adj = hetmatpy.matrix.metaedge_to_adjacency_matrix(
        graph, "DaG", dense_threshold=0
    )
    hetmat_adj = hetmatpy.matrix.metaedge_to_adjacency_matrix(
        hetmat, "DaG", dense_threshold=0
    )
    assert hetnet_adj[0] == hetmat_adj[0]  # row identifiers
    assert hetnet_adj[1] == hetmat_adj[1]  # column identifiers
    assert numpy.array_equal(hetnet_adj[2], hetmat_adj[2])  # adj matrices
