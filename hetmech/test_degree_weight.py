import hetio.readwrite
import pytest

from .degree_weight import dwwc


def test_disease_gene_example_dwwc():
    """
    Test the PC & DWWC computations in Figure 2D of Himmelstein & Baranzini
    (2015) PLOS Comp Bio. https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph

    # Compute GiGaD path count and DWWC matrices
    metapath = metagraph.metapath_from_abbrev('GiGaD')
    rows, cols, pc_matrix = dwwc(graph, metapath, damping=0)
    rows, cols, dwwc_matrix = dwwc(graph, metapath, damping=0.5)

    # Check row and column name assignment
    assert rows == ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    assert cols == ["Crohn's Disease", 'Multiple Sclerosis']

    # Check concordance with https://doi.org/10.1371/journal.pcbi.1004259.g002
    i = rows.index('IRF1')
    j = cols.index('Multiple Sclerosis')

    # Warning: the WC (walk count) and PC (path count) are only equivalent
    # because none of the GiGaD paths contain duplicate nodes. Since, GiGaD
    # contains duplicate metanodes, WC and PC are not gauranteed to be the
    # same. However, they happen to be equivalent for this example.
    assert pc_matrix[i, j] == 3
    assert dwwc_matrix[i, j] == pytest.approx(0.25 + 0.25 + 32**-0.5)
