import hetio.readwrite
import numpy
import pytest
from scipy import sparse

from .degree_weight import dwwc, dwpc_duplicated_metanode


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
    rows, cols, wc_matrix = dwwc(graph, metapath, damping=0)
    rows, cols, dwwc_matrix = dwwc(graph, metapath, damping=0.5)

    # Check row and column name assignment
    assert rows == ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1']
    assert cols == ["Crohn's Disease", 'Multiple Sclerosis']

    # Check concordance with https://doi.org/10.1371/journal.pcbi.1004259.g002
    i = rows.index('IRF1')
    j = cols.index('Multiple Sclerosis')

    # Warning: the WC (walk count) and PC (path count) are only equivalent
    # because none of the GiGaD paths contain duplicate nodes. Since, GiGaD
    # contains duplicate metanodes, WC and PC are not guaranteed to be the
    # same. However, they happen to be equivalent for this example.
    assert wc_matrix[i, j] == 3
    assert dwwc_matrix[i, j] == pytest.approx(0.25 + 0.25 + 32 ** -0.5)


def get_expected(metapath, threshold):
    # Dictionary with tuples of matrix and percent nonzero
    mat_dict = {
        'GiGaD': ([[0.25, 0.],
                   [0.35355339, 0.],
                   [0., 0.6767767],
                   [0.35355339, 0.],
                   [0., 0.35355339],
                   [0., 0.],
                   [0.35355339, 0.]], 0.429),
        'GaDaG': ([[0.25, 0.25, 0., 0.25, 0., 0.1767767, 0.],
                   [0.25, 0.25, 0., 0.25, 0., 0.1767767, 0.],
                   [0., 0., 0.5, 0., 0., 0.35355339, 0.],
                   [0.25, 0.25, 0., 0.25, 0., 0.1767767, 0.],
                   [0., 0., 0., 0., 0., 0., 0.],
                   [0.1767767, 0.1767767, 0.35355339, 0.1767767, 0., 0.375,
                    0.],
                   [0., 0., 0., 0., 0., 0., 0.]], 0.388),
        'GeTlD': ([[0., 0.],
                   [0., 0.],
                   [0., 0.70710678],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]], 0.071),
        'GiG': ([[0., 0., 0.35355339, 0., 0.70710678, 0., 0.],
                 [0., 0., 0.5, 0., 0., 0., 0.],
                 [0.35355339, 0.5, 0., 0.5, 0., 0., 0.5],
                 [0., 0., 0.5, 0., 0., 0., 0.],
                 [0.70710678, 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0.5, 0., 0., 0., 0.]], 0.204),
        'GaDaG_dwpc': ([[0., 0.25, 0., 0.25, 0., 0.1767767, 0.],
                        [0.25, 0., 0., 0.25, 0., 0.1767767, 0.],
                        [0., 0., 0., 0., 0., 0.35355339, 0.],
                        [0.25, 0.25, 0., 0., 0., 0.1767767, 0.],
                        [0., 0., 0., 0., 0., 0., 0.],
                        [0.1767767, 0.1767767, 0.35355339, 0.1767767, 0.,
                         0., 0.],
                        [0., 0., 0., 0., 0., 0., 0.]], 0.288)
    }
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    type_dict = {
        True: numpy.array,
        False: sparse.csc_matrix
    }
    row_names = node_dict[metapath[0]]
    col_names = node_dict[metapath[-1]]

    if metapath == 'GaDaG':
        dwwc_mat = mat_dict['GaDaG'][0]
        dwpc_mat = mat_dict['GaDaG_dwpc'][0]
        type_dwwc = type_dict[mat_dict['GaDaG'][1] >= threshold]
        type_dwpc = type_dict[mat_dict['GaDaG_dwpc'][1] >= threshold]
    else:
        dwwc_mat = dwpc_mat = mat_dict[metapath][0]
        type_dwwc = type_dict[mat_dict[metapath][1] >= threshold]
        type_dwpc = type_dict[mat_dict[metapath][1] >= threshold]

    dwwc_mat = type_dwwc(dwwc_mat)
    dwpc_mat = type_dwpc(dwpc_mat)

    return type_dwwc, dwwc_mat, type_dwpc, dwpc_mat, row_names, col_names


@pytest.mark.parametrize('thresh', (0, 0.25, 0.3, 0.5, 0.7, 1))
@pytest.mark.parametrize('m_path', ('GiGaD', 'GaDaG', 'GeTlD', 'GiG'))
def test_dwpc_duplicated_metanode(m_path, thresh):
    """
    Test the ability of dwpc_duplicated_metanode to convert dwpc_matrix to a
    dense array when the percent nonzero goes above the threshold.
    Checks output of dwpc_duplicated_metanode.
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph
    metapath = metagraph.metapath_from_abbrev(m_path)
    dup = metapath.get_nodes()[0]
    rows, cols, dwwc_mat = dwpc_duplicated_metanode(
        graph, metapath, damping=0.5, duplicate=None, sparse_threshold=thresh)
    rows, cols, dwpc_mat = dwpc_duplicated_metanode(
        graph, metapath, damping=0.5, duplicate=dup, sparse_threshold=thresh)

    exp_type_dwwc, exp_dwwc, exp_type_dwpc, exp_dwpc, exp_row, exp_col = \
        get_expected(m_path, thresh)

    if exp_type_dwwc == numpy.array:
        assert isinstance(dwwc_mat, numpy.ndarray)
    else:
        assert sparse.issparse(dwwc_mat)

    if exp_type_dwpc == numpy.array:
        assert isinstance(dwpc_mat, numpy.ndarray)
    else:
        assert sparse.issparse(dwpc_mat)

    # Test matrix, row, and column label output
    assert pytest.approx((dwwc_mat - exp_dwwc).sum(), 0)  # Assert approx equal
    assert pytest.approx((dwpc_mat - exp_dwpc).sum(), 0)  # Assert approx equal
    assert rows == exp_row
    assert cols == exp_col
