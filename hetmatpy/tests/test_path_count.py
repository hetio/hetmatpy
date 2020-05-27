import platform

import numpy
import pytest

import hetnetpy.pathtools
from hetmatpy.degree_weight import dwpc
from hetmatpy.testing import get_graph


def test_CbGpPWpGaD_traversal():
    """
    Test path counts and degree-weighted path counts for the CbGpPWpGaD
    metapath between bupropion and nicotine dependence. Expected values from
    the network traversal methods at https://git.io/vHBh2.
    """
    graph = get_graph('bupropion-subgraph')
    compound = 'DB01156'  # Bupropion
    disease = 'DOID:0050742'  # nicotine dependence
    metapath = graph.metagraph.metapath_from_abbrev('CbGpPWpGaD')
    rows, cols, pc_matrix = dwpc(graph, metapath, damping=0)
    rows, cols, dwpc_matrix = dwpc(graph, metapath, damping=0.4)
    i = rows.index(compound)
    j = cols.index(disease)
    assert pc_matrix[i, j] == 142
    assert dwpc_matrix[i, j] == pytest.approx(0.03287590886921623)


def test_CbGiGiGaD_traversal():
    """
    Test path counts and degree-weighted path counts for the CbGiGiGaD
    metapath between bupropion and nicotine dependence. These values are not
    intended to correspond to the values from the entire Hetionet v1.0. Hence,
    the expected values are generated using hetnetpy.pathtools.
    """
    graph = get_graph('bupropion-subgraph')
    compound = 'DB01156'  # Bupropion
    disease = 'DOID:0050742'  # nicotine dependence
    metapath = graph.metagraph.metapath_from_abbrev('CbGiGiGaD')
    paths = hetnetpy.pathtools.paths_between(
        graph,
        source=('Compound', compound),
        target=('Disease', disease),
        metapath=metapath,
        duplicates=False,
    )
    hetnetpy_dwpc = hetnetpy.pathtools.DWPC(paths, damping_exponent=0.4)

    rows, cols, pc_matrix = dwpc(graph, metapath, damping=0)
    rows, cols, dwpc_matrix = dwpc(graph, metapath, damping=0.4)
    i = rows.index(compound)
    j = cols.index(disease)

    assert pc_matrix[i, j] == len(paths)
    assert dwpc_matrix[i, j] == pytest.approx(hetnetpy_dwpc)


@pytest.mark.parametrize('metapath', [
    'CbGaD',
    'CbGbCtD',
    'CrCtD',
    'CtDrD',
    'CuGr>GuD',
    'CuG<rGuD',
    'CcSEcCtDpSpD',
])
@pytest.mark.parametrize('hetmat', [True, False])
def test_path_traversal(metapath, hetmat, tmpdir):
    """
    Test PC (path count) and DWPC (degree-weighted path count) computation
    on the random subgraph of Hetionet v1.0. Evaluates max path count
    compound-disease pair where errors are most likely to appear.
    """
    # Read graph
    if platform.system() == "Windows":
        pytest.xfail("path contains invalid character for Windows: >")
    graph = get_graph('random-subgraph')
    graph_or_hetmat = graph
    if hetmat:
        graph_or_hetmat = get_graph('random-subgraph', hetmat=hetmat, directory=tmpdir)
    metapath = graph.metagraph.metapath_from_abbrev(metapath)

    # Matrix computations
    rows, cols, pc_matrix = dwpc(graph_or_hetmat, metapath, damping=0)
    rows, cols, dwpc_matrix = dwpc(graph_or_hetmat, metapath, damping=0.4)

    # Find compound-disease pair with the max path count
    i, j = numpy.unravel_index(pc_matrix.argmax(), pc_matrix.shape)
    compound = rows[i]
    disease = cols[j]

    # hetnetpy.pathtools computations
    paths = hetnetpy.pathtools.paths_between(
        graph,
        source=('Compound', compound),
        target=('Disease', disease),
        metapath=metapath,
        duplicates=False,
    )
    hetnetpy_dwpc = hetnetpy.pathtools.DWPC(paths, damping_exponent=0.4)

    # Check matrix values match hetnetpy.pathtools
    assert pc_matrix[i, j] == len(paths)
    assert dwpc_matrix[i, j] == pytest.approx(hetnetpy_dwpc)
