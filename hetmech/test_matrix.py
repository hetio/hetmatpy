import hetio.readwrite
from .matrix import metaedge_to_adjacency_matrix
import numpy as np


def test_metaedge_to_adjacency_matrix():
    """
    Test the functionality of metaedge_to_adjacency_matrix in generating
    numpy arrays. Uses same test data as in test_degree_weight.py
    Figure 2D of Himmelstein & Baranzini (2015) PLOS Comp Bio.
    https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)

    # Verify GiG matrix
    gig_rows = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    gig_cols = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']

    row_names, col_names, adj_mat = metaedge_to_adjacency_matrix(graph, 'GiG')
    assert np.array_equal(row_names, gig_rows)
    assert np.array_equal(col_names, gig_cols)
    assert np.array_equal(adj_mat, [[0, 0, 1, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [1, 1, 0, 1, 0, 0, 1],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0]])

    # Verify GaD matrix
    gad_rows = ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    gad_cols = ["Crohn's Disease", 'Multiple Sclerosis']

    row_names, col_names, adj_mat = metaedge_to_adjacency_matrix(graph, 'GaD')
    assert np.array_equal(row_names, gad_rows)
    assert np.array_equal(col_names, gad_cols)
    assert np.array_equal(adj_mat, [[0, 1],
                                    [0, 1],
                                    [1, 0],
                                    [0, 1],
                                    [0, 0],
                                    [1, 1],
                                    [0, 0]])

    # Verify DlT matrix
    dlt_rows = ["Crohn's Disease", 'Multiple Sclerosis']
    dlt_cols = ['Leukocyte', 'Lung']

    row_names, col_names, adj_mat = metaedge_to_adjacency_matrix(graph, 'DlT')
    assert np.array_equal(row_names, dlt_rows)
    assert np.array_equal(col_names, dlt_cols)
    assert np.array_equal(adj_mat, [[0, 0],
                                    [1, 0]])

    # Verify TlD matrix
    tld_rows = ['Leukocyte', 'Lung']
    tld_cols = ["Crohn's Disease", 'Multiple Sclerosis']

    row_names, col_names, adj_mat = metaedge_to_adjacency_matrix(graph, "TlD")
    assert np.array_equal(row_names, tld_rows)
    assert np.array_equal(col_names, tld_cols)
    assert np.array_equal(adj_mat, [[0, 1],
                                    [0, 0]])
