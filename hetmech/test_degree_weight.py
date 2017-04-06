import hetio.readwrite
import pytest

from .degree_weight import dwwc
from .matrix import get_node_to_position


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
    pc_matrix = dwwc(graph, metapath, damping=0)
    dwwc_matrix = dwwc(graph, metapath, damping=0.5)

    # Check concordance with https://doi.org/10.1371/journal.pcbi.1004259.g002
    gene_index = get_node_to_position(graph, 'Gene')
    disease_index = get_node_to_position(graph, 'Disease')
    i = gene_index[graph.node_dict['Gene', 'IRF1']]
    j = disease_index[graph.node_dict['Disease', 'Multiple Sclerosis']]

    # Warning: the WC (walk count) and PC (path count) are only equivalent
    # because none of the GiGaD paths contain duplicate nodes. Since, GiGaD
    # contains duplicate metanodes, WC and PC are not gauranteed to be the
    # same. However, they happen to be equivalent for this example.
    assert pc_matrix[i, j] == 3
    assert dwwc_matrix[i, j] == pytest.approx(0.25 + 0.25 + 32**-0.5)
