import hetio.pathtools
import hetio.readwrite
import numpy
import pytest

from hetmech.degree_weight import dwpc


def get_bupropion_subgraph():
    """
    Read the bupropion and nicotine dependence Hetionet v1.0 subgraph.
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '30c6dbb18a17c05d71cb909cf57af7372e4d4908',
        'test/data/bupropion-CbGpPWpGaD-subgraph.json.xz',
    )
    return hetio.readwrite.read_graph(url)


def get_random_subgraph():
    """
    Read the bupropion and nicotine dependence Hetionet v1.0 subgraph.
    """
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '30c6dbb18a17c05d71cb909cf57af7372e4d4908',
        'test/data/random-subgraph.json.xz',
    )
    return hetio.readwrite.read_graph(url)


def test_CbGpPWpGaD_traversal():
    """
    Test path counts and degree-weighted path counts for the CbGpPWpGaD
    metapath between bupropion and nicotine dependence. Expected values from
    the network traversal methods at https://git.io/vHBh2.
    """
    graph = get_bupropion_subgraph()
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
    the expected values are generated using hetio.pathtools.
    """
    graph = get_bupropion_subgraph()
    compound = 'DB01156'  # Bupropion
    disease = 'DOID:0050742'  # nicotine dependence
    metapath = graph.metagraph.metapath_from_abbrev('CbGiGiGaD')
    paths = hetio.pathtools.paths_between(
        graph,
        source=('Compound', compound),
        target=('Disease', disease),
        metapath=metapath,
        duplicates=False,
    )
    hetio_dwpc = hetio.pathtools.DWPC(paths, damping_exponent=0.4)

    rows, cols, pc_matrix = dwpc(graph, metapath, damping=0)
    rows, cols, dwpc_matrix = dwpc(graph, metapath, damping=0.4)
    i = rows.index(compound)
    j = cols.index(disease)

    assert pc_matrix[i, j] == len(paths)
    assert dwpc_matrix[i, j] == pytest.approx(hetio_dwpc)


@pytest.mark.parametrize('metapath', [
                                      'CbGaD',
                                      'CbGbCtD',
                                      'CrCtD',
                                      'CtDrD',
                                      'CuGr>GuD',
                                      'CuG<rGuD',
                                      'CcSEcCtDpSpD',
                                     ])
def test_path_traversal(metapath):
    """
    Test PC (path count) and DWPC (degree-weighted path count) computation
    on the random subgraph of Hetionet v1.0. Evaluates max path count
    compound-disease pair where errors are most likely to appear.
    """
    # Read graph
    graph = get_random_subgraph()
    metapath = graph.metagraph.metapath_from_abbrev(metapath)

    # Matrix computations
    rows, cols, pc_matrix = dwpc(graph, metapath, damping=0)
    rows, cols, dwpc_matrix = dwpc(graph, metapath, damping=0.4)

    # Find compound-disease pair with the max path count
    i, j = numpy.unravel_index(pc_matrix.argmax(), pc_matrix.shape)
    compound = rows[i]
    disease = cols[j]

    # hetio.pathtools computations
    paths = hetio.pathtools.paths_between(
        graph,
        source=('Compound', compound),
        target=('Disease', disease),
        metapath=metapath,
        duplicates=False,
    )
    hetio_dwpc = hetio.pathtools.DWPC(paths, damping_exponent=0.4)

    # Check matrix values match hetio.pathtools
    assert pc_matrix[i, j] == len(paths)
    assert dwpc_matrix[i, j] == pytest.approx(hetio_dwpc)
