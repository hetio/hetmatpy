import hetnetpy.readwrite
import numpy
import pytest
from scipy import sparse

from hetmatpy.degree_weight import (
    _dwpc_approx,
    _dwpc_baab,
    _dwpc_baba,
    _dwpc_general_case,
    _dwpc_short_repeat,
    categorize,
    dwpc,
    dwwc,
    dwwc_chain,
    dwwc_recursive,
    dwwc_sequential,
    get_segments,
)
from hetmatpy.testing import get_graph


@pytest.mark.parametrize(
    "dwwc_method",
    [
        None,
        dwwc_sequential,
        dwwc_recursive,
        dwwc_chain,
    ],
)
def test_disease_gene_example_dwwc(dwwc_method):
    """
    Test the PC & DWWC computations in Figure 2D of Himmelstein & Baranzini
    (2015) PLOS Comp Bio. https://doi.org/10.1371/journal.pcbi.1004259.g002
    """
    graph = get_graph("disease-gene-example")
    metagraph = graph.metagraph

    # Compute GiGaD path count and DWWC matrices
    metapath = metagraph.metapath_from_abbrev("GiGaD")
    rows, cols, wc_matrix = dwwc(graph, metapath, damping=0, dwwc_method=dwwc_method)
    rows, cols, dwwc_matrix = dwwc(
        graph, metapath, damping=0.5, dwwc_method=dwwc_method
    )

    # Check row and column name assignment
    assert rows == ["CXCR4", "IL2RA", "IRF1", "IRF8", "ITCH", "STAT3", "SUMO1"]
    assert cols == ["Crohn's Disease", "Multiple Sclerosis"]

    # Check concordance with https://doi.org/10.1371/journal.pcbi.1004259.g002
    i = rows.index("IRF1")
    j = cols.index("Multiple Sclerosis")

    # Warning: the WC (walk count) and PC (path count) are only equivalent
    # because none of the GiGaD paths contain duplicate nodes. Since, GiGaD
    # contains duplicate metanodes, WC and PC are not guaranteed to be the
    # same. However, they happen to be equivalent for this example.
    assert wc_matrix[i, j] == 3
    assert dwwc_matrix[i, j] == pytest.approx(0.25 + 0.25 + 32**-0.5)


def get_nodes(metapath):
    node_dict = {
        "G": ["CXCR4", "IL2RA", "IRF1", "IRF8", "ITCH", "STAT3", "SUMO1"],
        "D": ["Crohn's Disease", "Multiple Sclerosis"],
        "T": ["Leukocyte", "Lung"],
    }
    exp_row = node_dict[metapath[0]]
    exp_col = node_dict[metapath[-1]]
    return exp_row, exp_col


@pytest.mark.parametrize(
    "metapath,expected,path_type",
    [
        ("DaGeT", [[0.5, 0.5], [0, 0]], 0),
        ("DlTeG", [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0.70710678, 0, 0, 0, 0]], 0),
        ("GeTlD", [[0, 0], [0, 0], [0, 0.70710678], [0, 0], [0, 0], [0, 0], [0, 0]], 0),
        (
            "GaDlT",
            [[0.5, 0], [0.5, 0], [0, 0], [0.5, 0], [0, 0], [0.35355339, 0], [0, 0]],
            0,
        ),
        ("TeGaD", [[0.5, 0], [0.5, 0]], 0),
        ("TlDaG", [[0.5, 0.5, 0, 0.5, 0, 0.35355339, 0], [0, 0, 0, 0, 0, 0, 0]], 0),
        (
            "GiG",
            [
                [0.0, 0.0, 0.35355339, 0.0, 0.70710678, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.35355339, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            ],
            1,
        ),
        (
            "GaDaG",
            [
                [0.0, 0.25, 0.0, 0.25, 0.0, 0.1767767, 0.0],
                [0.25, 0.0, 0.0, 0.25, 0.0, 0.1767767, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.35355339, 0.0],
                [0.25, 0.25, 0.0, 0.0, 0.0, 0.1767767, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1767767, 0.1767767, 0.35355339, 0.1767767, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            1,
        ),
        (
            "GiGiG",
            [
                [0, 0.1767767, 0, 0.1767767, 0, 0, 0.1767767],
                [0.1767767, 0, 0, 0.25, 0, 0, 0.25],
                [0, 0, 0, 0, 0.25, 0, 0],
                [0.1767767, 0.25, 0, 0, 0, 0, 0.25],
                [0, 0, 0.25, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0.1767767, 0.25, 0, 0.25, 0, 0, 0],
            ],
            1,
        ),
    ],
)
def test_no_and_short_repeat(metapath, expected, path_type):
    exp_row, exp_col = get_nodes(metapath)
    graph = get_graph("disease-gene-example")
    metapath = graph.metagraph.metapath_from_abbrev(metapath)
    func_dict = {0: dwwc, 1: _dwpc_short_repeat}

    row, col, dwpc_matrix = func_dict[path_type](graph, metapath, damping=0.5)

    expected = numpy.array(expected, dtype=numpy.float64)
    assert abs(dwpc_matrix - expected).max() == pytest.approx(0, abs=1e-7)
    assert row == exp_row
    assert col == exp_col


@pytest.mark.parametrize(
    "metapath,exp_row,exp_col,exp_data,shape",
    [
        (
            "DrDaGiG",
            [1, 9, 16, 26, 43, 68, 17, 21, 25, 29, 36, 38, 62, 73],
            [21, 21, 21, 21, 21, 21, 42, 42, 42, 42, 42, 42, 42, 42],
            [
                0.111803398875,
                0.380039827693,
                0.102062072616,
                0.288675134595,
                0.204124145232,
                0.144337567297,
                0.037688918072,
                0.058925565098,
                0.044194173824,
                0.055901699437,
                0.047245559126,
                0.055901699437,
                0.039528470752,
                0.05103103630,
            ],
            (104, 105),
        ),
        (
            "CrCpDrD",
            [
                3,
                44,
                49,
                44,
                44,
                22,
                84,
                22,
                84,
                44,
                44,
                44,
                3,
                22,
                49,
                84,
                40,
                51,
                84,
                40,
                51,
                84,
                40,
                51,
                84,
                44,
                44,
                44,
                44,
                44,
                44,
                44,
                44,
                44,
            ],
            [
                2,
                2,
                2,
                13,
                14,
                16,
                16,
                30,
                30,
                32,
                33,
                37,
                45,
                45,
                45,
                45,
                51,
                51,
                51,
                56,
                56,
                56,
                57,
                57,
                57,
                61,
                74,
                81,
                85,
                88,
                89,
                93,
                97,
                99,
            ],
            [
                0.22360679775,
                0.115470053838,
                0.22360679775,
                0.105409255339,
                0.0645497224368,
                0.068041381744,
                0.0833333333333,
                0.0833333333333,
                0.102062072616,
                0.105409255339,
                0.0645497224368,
                0.0873962324984,
                0.22360679775,
                0.07453559925,
                0.22360679775,
                0.0912870929175,
                0.0589255650989,
                0.0589255650989,
                0.0589255650989,
                0.0710669054519,
                0.0710669054519,
                0.0710669054519,
                0.0710669054519,
                0.0710669054519,
                0.0710669054519,
                0.0645497224368,
                0.107038087531,
                0.1490711985,
                0.182574185835,
                0.0873962324984,
                0.0589255650989,
                0.0833333333333,
                0.0926977029746,
                0.0416666666667,
            ],
            (103, 104),
        ),
        ("CrCpDrDaGiG", [0], [0], [0], (103, 105)),
        (
            "CrCrCpDrDrD",
            [
                22,
                22,
                22,
                22,
                22,
                22,
                22,
                22,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                40,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                51,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                61,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
                84,
            ],
            [
                1,
                2,
                13,
                16,
                30,
                32,
                43,
                85,
                1,
                2,
                13,
                16,
                30,
                32,
                43,
                85,
                5,
                7,
                8,
                17,
                18,
                20,
                24,
                25,
                29,
                34,
                35,
                36,
                38,
                48,
                49,
                50,
                53,
                56,
                57,
                59,
                62,
                63,
                65,
                70,
                72,
                73,
                75,
                82,
                83,
                94,
                95,
                96,
                5,
                7,
                8,
                17,
                18,
                20,
                24,
                25,
                29,
                34,
                35,
                36,
                38,
                48,
                49,
                50,
                53,
                56,
                57,
                59,
                62,
                63,
                65,
                70,
                72,
                73,
                75,
                82,
                83,
                94,
                95,
                96,
                1,
                2,
                5,
                7,
                8,
                13,
                16,
                17,
                18,
                20,
                24,
                25,
                29,
                30,
                32,
                34,
                35,
                36,
                38,
                43,
                48,
                49,
                50,
                53,
                56,
                57,
                59,
                62,
                63,
                65,
                70,
                72,
                73,
                75,
                82,
                83,
                85,
                94,
                95,
                96,
            ],
            [
                0.00621129,
                0.00745355,
                0.02097942,
                0.00850517,
                0.00694444,
                0.02097942,
                0.00694444,
                0.01178511,
                0.00507150,
                0.00608580,
                0.01712962,
                0.00694444,
                0.00567011,
                0.01712962,
                0.00567011,
                0.00962250,
                0.00260416,
                0.00329403,
                0.00357124,
                0.00157037,
                0.00714249,
                0.00535686,
                0.00204287,
                0.00184142,
                0.00338798,
                0.00368284,
                0.00276627,
                0.00483193,
                0.00338798,
                0.00222084,
                0.00357124,
                0.00323031,
                0.00404941,
                0.00323031,
                0.00323031,
                0.00260416,
                0.00404268,
                0.00297145,
                0.00404941,
                0.00309279,
                0.00173611,
                0.00212629,
                0.00479132,
                0.00196856,
                0.00329403,
                0.00479132,
                0.00437386,
                0.00245523,
                0.00368284,
                0.00465847,
                0.00505050,
                0.00222084,
                0.01010101,
                0.00757575,
                0.00288906,
                0.00260416,
                0.00479132,
                0.00520833,
                0.00391210,
                0.00683338,
                0.00479132,
                0.00314074,
                0.00505050,
                0.00456835,
                0.00572673,
                0.00456835,
                0.00456835,
                0.00368284,
                0.00571721,
                0.00420227,
                0.00572673,
                0.00437386,
                0.00245523,
                0.00300703,
                0.00677596,
                0.00278397,
                0.00465847,
                0.00677596,
                0.00618558,
                0.00347222,
                0.00507150,
                0.00608580,
                0.00260416,
                0.00329403,
                0.00357124,
                0.01712962,
                0.00694444,
                0.00157037,
                0.00714249,
                0.00535686,
                0.00204287,
                0.00184142,
                0.00338798,
                0.00567011,
                0.01712962,
                0.00368284,
                0.00276627,
                0.00483193,
                0.00338798,
                0.00567011,
                0.00222084,
                0.00357124,
                0.00323031,
                0.00404941,
                0.00323031,
                0.00323031,
                0.00260416,
                0.00404268,
                0.00297145,
                0.00404941,
                0.00309279,
                0.00173611,
                0.00212629,
                0.00479132,
                0.00196856,
                0.00329403,
                0.00962250,
                0.00479132,
                0.00437386,
                0.00245523,
            ],
            (103, 104),
        ),
    ],
)
def test_disjoint_dwpc(metapath, exp_row, exp_col, exp_data, shape):
    graph = get_graph("random-subgraph")
    metapath = graph.metagraph.metapath_from_abbrev(metapath)

    row, col, dwpc_matrix = dwpc(graph, metapath)

    # expected = numpy.array(expected, dtype=numpy.float64)
    expected = sparse.coo_matrix((exp_data, (exp_row, exp_col)), shape=shape)
    assert abs(dwpc_matrix - expected).max() == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize(
    "metapath,expected",
    [
        ("DaGiGaD", [[0.0, 0.47855339], [0.47855339, 0.0]]),
        ("TeGiGeT", [[0, 0], [0, 0]]),
        ("DaGiGeTlD", [[0, 0], [0, 0]]),
        ("DaGeTeGaD", [[0, 0], [0, 0]]),
        ("TlDaGiGeT", [[0.0, 0.47855339], [0.0, 0.0]]),
        ("DaGiGaDlT", [[0.47855339, 0], [0, 0]]),
    ],
)
def test__dwpc_baab(metapath, expected):
    exp_row, exp_col = get_nodes(metapath)
    graph = get_graph("disease-gene-example")
    metapath = graph.metagraph.metapath_from_abbrev(metapath)

    row, col, dwpc_matrix = _dwpc_baab(graph, metapath, damping=0.5, dense_threshold=1)

    expected = numpy.array(expected, dtype=numpy.float64)

    assert abs(dwpc_matrix - expected).max() == pytest.approx(0, abs=1e-7)
    assert exp_row == row
    assert exp_col == col


def get_baba_matrices(metapath):
    node_dict = {
        "G": ["CXCR4", "IL2RA", "IRF1", "IRF8", "ITCH", "STAT3", "SUMO1"],
        "D": ["Crohn's Disease", "Multiple Sclerosis"],
        "T": ["Leukocyte", "Lung"],
    }
    edge_dict = {
        0: [
            [0.08838835, 0],
            [0.08838835, 0],
            [0, 0.125],
            [0.08838835, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],
        1: [[0, 0], [0, 0]],
        2: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        3: [
            [0.25, 0.0],
            [0.25, 0.0],
            [0.0, 0.0],
            [0.25, 0.0],
            [0.0, 0.0],
            [0.1767767, 0.0],
            [0.0, 0.0],
        ],
        4: [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.125, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        5: [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.25],
            [0.0, 0.0],
        ],
        6: [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.125, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    }
    mat_dict = {
        "GaDaGaD": (0, 0),
        "DaGaDaG": (0, 1),
        "DlTlDlT": (1, 0),
        "TlDlTlD": (1, 1),
        "GeTeGeT": (2, 0),
        "TeGeTeG": (2, 1),
        "GaDlTeGaD": (3, 0),
        "DaGeTlDaG": (3, 1),
        "GeTlDaGaD": (4, 0),
        "DaGaDlTeG": (4, 1),
        "GaDaGeTlD": (5, 0),
        "DlTeGaDaG": (5, 1),
        "TlDaGaDaG": (6, 1),
    }
    first = node_dict[metapath[0]]
    last = node_dict[metapath[-1]]
    edge = mat_dict[metapath]
    adj = numpy.array(edge_dict[edge[0]], dtype=numpy.float64)
    if edge[1]:
        adj = adj.transpose()
    return first, last, adj


@pytest.mark.parametrize(
    "m_path",
    (
        "GaDaGaD",
        "DaGaDaG",
        "DlTlDlT",
        "TlDlTlD",
        "GeTeGeT",
        "TeGeTeG",
        "GaDlTeGaD",
        "GeTlDaGaD",
        "GaDaGeTlD",
        "TlDaGaDaG",
    ),
)
def test__dwpc_baba(m_path):
    graph = get_graph("disease-gene-example")
    metagraph = graph.metagraph
    metapath = metagraph.metapath_from_abbrev(m_path)

    row_sol, col_sol, adj_sol = get_baba_matrices(m_path)
    row, col, dwpc_matrix = _dwpc_baba(graph, metapath, damping=0.5, dense_threshold=0)

    assert row_sol == row
    assert col_sol == col
    assert abs(adj_sol - dwpc_matrix).max() == pytest.approx(0, abs=1e-8)


def get_general_solutions(length):
    genes = ["CXCR4", "IL2RA", "IRF1", "IRF8", "ITCH", "STAT3", "SUMO1"]
    mat_dict = {
        0: [
            [0, 0, 0.35355339, 0, 0.70710678, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0],
            [0.35355339, 0.5, 0, 0.5, 0, 0, 0.5],
            [0, 0, 0.5, 0, 0, 0, 0],
            [0.70710678, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0, 0],
        ],
        1: [
            [0, 0.1767767, 0, 0.1767767, 0, 0, 0.1767767],
            [0.1767767, 0, 0, 0.25, 0, 0, 0.25],
            [0, 0, 0, 0, 0.25, 0, 0],
            [0.1767767, 0.25, 0, 0, 0, 0, 0.25],
            [0, 0, 0.25, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0.1767767, 0.25, 0, 0.25, 0, 0, 0],
        ],
        2: [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0],
            [0, 0.125, 0, 0.125, 0, 0, 0.125],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.125, 0, 0],
        ],
        3: numpy.zeros((7, 7)),
        4: numpy.zeros((7, 7)),
        5: numpy.zeros((7, 7)),
    }
    return genes, genes, mat_dict[length]


@pytest.mark.parametrize("length", list(range(6)))
def test__dwpc_general_case(length):
    """
    Test the functionality of dwpc_same_metanode to find DWPC
    within a metapath (segment) of metanode and metaedge repeats.
    """
    graph = get_graph("disease-gene-example")
    metagraph = graph.metagraph
    m_path = "GiG" + length * "iG"
    metapath = metagraph.metapath_from_abbrev(m_path)
    rows, cols, dwpc_mat = _dwpc_general_case(graph, metapath, damping=0.5)
    exp_row, exp_col, exp_dwpc = get_general_solutions(length)

    # Test matrix, row, and column label output
    assert abs(dwpc_mat - exp_dwpc).max() == pytest.approx(0, abs=1e-7)
    assert rows == exp_row
    assert cols == exp_col


@pytest.mark.parametrize(
    "metapath,solution",
    [
        ("GiG", "short_repeat"),
        ("GiGiGiG", "four_repeat"),
        ("G" + 10 * "iG", "long_repeat"),
        ("GiGiGcGcG", "long_repeat"),  # iicc
        ("GiGcGcGiG", "long_repeat"),  # icci
        ("GcGiGcGaDrD", "disjoint"),  # cicDD
        ("GcGiGaDrDrD", "disjoint"),  # ciDDD
        ("CpDaG", "no_repeats"),  # ABC
        ("DaGiGaDaG", "other"),  # ABBAB
        ("DaGiGbC", "short_repeat"),  # ABBC
        ("DaGiGaD", "BAAB"),  # ABBA
        ("GeAlDlAeG", "BAAB"),  # ABCBA
        ("CbGaDrDaGeA", "BAAB"),  # ABCCBD
        ("AlDlAlD", "BABA"),  # ABAB
        ("CrCbGbCbG", "other"),  # BBABA
        ("CbGiGbCrC", "other"),
        ("CbGiGiGbC", "BAAB"),
        ("CbGbCbGbC", "other"),
        ("CrCbGiGbC", "other"),
        ("CrCbGbCbG", "other"),
        ("CbGaDaGeAlD", "BABA"),  # ABCBDC
        ("AlDaGiG", "short_repeat"),  # ABCC
        ("AeGaDaGiG", "short_repeat"),  # ABCB
        ("CbGaDpCbGaD", "other"),  # ABCABC
        ("DaGiGiGiGiGaD", "other"),  # ABBBBBA
        ("CbGaDrDaGbC", "BAAB"),  # ABCCBA
        ("DlAuGcGpBPpGaDlA", "other"),  # ABCCDCAB
        ("CrCbGiGaDrD", "disjoint"),  # AABBCC
        ("SEcCpDaGeAeGaDtC", "BAAB"),
        ("CbGiGiGbC", "BAAB"),
        ("CbGbCbGbC", "other"),
    ],
)  # ABABA
def test_categorize(metapath, solution):
    url = "https://github.com/hetio/hetnetpy/raw/{}/{}".format(
        "9dc747b8fc4e23ef3437829ffde4d047f2e1bdde",
        "test/data/hetionet-v1.0-metagraph.json",
    )
    metagraph = hetnetpy.readwrite.read_metagraph(url)
    metapath = metagraph.metapath_from_abbrev(metapath)
    assert categorize(metapath) == solution


@pytest.mark.parametrize(
    "metapath,solution",
    [
        ("AeGiGaDaG", "[AeG, GiGaDaG]"),  # short_repeat
        ("AeGaDaGiG", "[AeG, GaDaGiG]"),  # short_repeat other direction
        ("CpDrDdGdD", "[CpD, DrDdGdD]"),
        ("AeGiGeAlD", "[AeG, GiG, GeA, AlD]"),  # BAABC
        ("AeGiGaDlA", "[AeG, GiG, GaDlA]"),
        ("DaGaDaG", "[DaG, GaD, DaG]"),  # BABA
        ("CbGeAlDaGbC", "[CbG, GeAlDaG, GbC]"),
        ("SEcCpDaGeAeGaDtC", "[SEcC, CpD, DaG, GeAeG, GaD, DtC]"),
        ("DlAeGaDaG", "[DlAeG, GaD, DaG]"),  # BCABA
        ("GaDlAeGaD", "[GaD, DlAeG, GaD]"),  # BACBA
        ("GiGiG", "[GiGiG]"),  # short_repeat
        ("GiGiGiG", "[GiG, GiG, GiG]"),  # four_repeat
        ("CrCbGiGiGaDrDlA", "[CrC, CbG, GiGiG, GaD, DrD, DlA]"),
        ("CrCrCbGiGeAlDrD", "[CrCrC, CbG, GiG, GeAlD, DrD]"),
        ("SEcCrCrCbGiGeAlDrDpS", "[SEcC, CrCrC, CbG, GiG, GeAlD, DrD, DpS]"),
        ("SEcCrCrCrC", "[SEcC, CrC, CrC, CrC]"),
        ("GiGaDaG", "[GiGaDaG]"),
        ("CrCbGiGbC", "[CrC, CbG, GiG, GbC]"),  # OTHER
        ("GbCpDrDaG", "[GbCpD, DrD, DaG]"),
        ("CbGiGiGbC", "[CbG, GiGiG, GbC]"),
        ("CbGiGiGiGiGbC", "[CbG, GiGiGiGiG, GbC]"),  # OTHER
        ("CbGaDaGiGiGbCrC", "[CbG, GaDaGiGiG, GbC, CrC]"),  # OTHER
        ("CbGiGiGiGbCbG", "[CbG, GiGiGiG, GbC, CbG]"),
        ("CbGiGiGbCpD", "[CbG, GiGiG, GbC, CpD]"),
        ("CbGaDaGaDpC", "[CbG, GaDaGaD, DpC]"),
        ("GaDaGaD", "[GaD, DaG, GaD]"),
        ("CrCbGaDrDaG", "[CrC, CbG, GaDrDaG]"),
        ("CrCbGaDaGaD", "[CrC, CbG, GaDaGaD]"),
        ("DlAeGiGaDlA", "[DlA, AeGiGaD, DlA]"),
    ],
)
def test_get_segments(metapath, solution):
    url = "https://github.com/hetio/hetnetpy/raw/{}/{}".format(
        "9dc747b8fc4e23ef3437829ffde4d047f2e1bdde",
        "test/data/hetionet-v1.0-metagraph.json",
    )
    metagraph = hetnetpy.readwrite.read_metagraph(url)
    metapath = metagraph.metapath_from_abbrev(metapath)
    output = str(get_segments(metagraph, metapath))
    assert output == solution


@pytest.mark.parametrize("dense_threshold", [0, 1])
@pytest.mark.parametrize(
    "metapath,expected",
    [
        (
            "DaGiGiG",
            [
                [0.0, 0.0, 0.0, 0.0, 0.1767767, 0.0, 0.0],
                [0.1767767, 0.21338835, 0.0, 0.21338835, 0.0, 0.0, 0.33838835],
            ],
        ),
        ("DaGiGiGaD", [[0, 0], [0, 0]]),
        ("DaGiGaD", [[0.0, 0.47855339], [0.47855339, 0.0]]),
        ("TeGiGeT", [[0, 0], [0, 0]]),
        ("DaGiGeTlD", [[0, 0], [0, 0]]),
        ("DaGeTeGaD", [[0, 0], [0, 0]]),
        ("TlDaGiGeT", [[0.0, 0.47855339], [0.0, 0.0]]),
        ("DaGeT", [[0.5, 0.5], [0, 0]]),
        ("DlTeG", [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0.70710678, 0, 0, 0, 0]]),
        ("GeTlD", [[0, 0], [0, 0], [0, 0.70710678], [0, 0], [0, 0], [0, 0], [0, 0]]),
        (
            "GaDlT",
            [[0.5, 0], [0.5, 0], [0, 0], [0.5, 0], [0, 0], [0.35355339, 0], [0, 0]],
        ),
        ("TeGaD", [[0.5, 0], [0.5, 0]]),
        ("TlDaG", [[0.5, 0.5, 0, 0.5, 0, 0.35355339, 0], [0, 0, 0, 0, 0, 0, 0]]),
        (
            "GiG",
            [
                [0.0, 0.0, 0.35355339, 0.0, 0.70710678, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.35355339, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
        (
            "GaDaG",
            [
                [0.0, 0.25, 0.0, 0.25, 0.0, 0.1767767, 0.0],
                [0.25, 0.0, 0.0, 0.25, 0.0, 0.1767767, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.35355339, 0.0],
                [0.25, 0.25, 0.0, 0.0, 0.0, 0.1767767, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1767767, 0.1767767, 0.35355339, 0.1767767, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
        (
            "GiGiG",
            [
                [0, 0.1767767, 0, 0.1767767, 0, 0, 0.1767767],
                [0.1767767, 0, 0, 0.25, 0, 0, 0.25],
                [0, 0, 0, 0, 0.25, 0, 0],
                [0.1767767, 0.25, 0, 0, 0, 0, 0.25],
                [0, 0, 0.25, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0.1767767, 0.25, 0, 0.25, 0, 0, 0],
            ],
        ),
        (
            "GiGiGiG",
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.125, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0.0, 0, 0, 0.125, 0, 0],
                [0, 0.125, 0, 0.125, 0, 0, 0.125],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.125, 0, 0],
            ],
        ),
        (
            "GaDaGaD",
            [
                [0.08838835, 0],  # BABA
                [0.08838835, 0],
                [0, 0.125],
                [0.08838835, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
        ),
        ("DlTlDlT", [[0, 0], [0, 0]]),  # BABA
        ("TlDlTlD", [[0, 0], [0, 0]]),  # BABA
        ("GeTeGeT", [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),  # BABA
        (
            "GaDlTeGaD",
            [
                [0.25, 0.0],  # BA C BA
                [0.25, 0.0],
                [0.0, 0.0],
                [0.25, 0.0],
                [0.0, 0.0],
                [0.1767767, 0.0],
                [0.0, 0.0],
            ],
        ),
        (
            "GeTlDaGaD",
            [
                [0.0, 0.0],  # B C ABA
                [0.0, 0.0],
                [0.125, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ),
        (
            "GaDaGeTlD",
            [
                [0.0, 0.0],  # BAB C A
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.25],
                [0.0, 0.0],
            ],
        ),
        ("TlDaGaDaGeT", [[0, 0.0883883476], [0, 0]]),  # C BABA C
        ("TlDaGiGaDlT", [[0, 0], [0, 0]]),  # C BAAB C
        ("TeGiGaDlTlD", [[0, 0], [0, 0]]),
        ("TeGiGaD", [[0.0, 0.47855339], [0.0, 0.47855339]]),
        (
            "TeGaDaG",
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
            ],
        ),
    ],
)
def test_dwpc(metapath, expected, dense_threshold):
    if expected is not None:
        expected = numpy.array(expected, dtype=numpy.float64)

    graph = get_graph("disease-gene-example")
    metapath = graph.metagraph.metapath_from_abbrev(metapath)
    if expected is None:
        with pytest.raises(Exception):
            dwpc(graph, metapath, damping=0.5, dense_threshold=dense_threshold)
    else:
        row, col, dwpc_matrix = dwpc(
            graph, metapath, damping=0.5, dense_threshold=dense_threshold
        )
        assert abs(expected - dwpc_matrix).max() == pytest.approx(0, abs=1e-7)
        if dense_threshold == 1:
            assert sparse.issparse(dwpc_matrix)
        else:
            assert not sparse.issparse(dwpc_matrix)


@pytest.mark.parametrize(
    "metapath,dtype",
    [
        ("TeGaDaG", numpy.float64),
        ("TeGaDaG", numpy.float32),
        # ('TeGaDaG', numpy.float16),  # fails due to https://github.com/scipy/scipy/issues/8903
    ],
)
@pytest.mark.parametrize(
    "dwwc_method",
    [
        None,
        dwwc_sequential,
        dwwc_recursive,
        dwwc_chain,
    ],
)
def test_dtype(metapath, dtype, dwwc_method):
    graph = get_graph("disease-gene-example")
    metapath = graph.metagraph.metapath_from_abbrev(metapath)
    rows, cols, dwpc_matrix = dwpc(
        graph, metapath, dtype=dtype, dwwc_method=dwwc_method
    )
    assert dwpc_matrix.dtype == dtype


@pytest.mark.parametrize(
    "metapath,relative",
    [
        ("DrDaGiG", "equal"),
        ("DaGiGaD", "equal"),
        ("DaGaDrDaGaD", "not_equal"),
        ("CrCpDrD", "equal"),
    ],
)
def test_dwpc_approx(metapath, relative):
    graph = get_graph("random-subgraph")
    metapath = graph.metagraph.metapath_from_abbrev(metapath)
    rows, cols, dwpc_matrix = dwpc(graph, metapath)
    rows, cols, dwpc_approx = _dwpc_approx(graph, metapath)
    rows, cols, dwwc_matrix = dwwc(graph, metapath)
    if relative == "equal":
        assert abs(dwpc_approx - dwpc_matrix).max() == pytest.approx(0, abs=1e-7)
    else:
        assert numpy.sum(dwpc_approx - dwpc_matrix) >= 0
    assert abs(dwwc_matrix - dwpc_approx).max() >= 0
