import hetio.readwrite
import numpy
import pytest

from .dwpc import dwpc_baab


@pytest.mark.parametrize('metapath,expected', [
    ('DaGiGaD', [[0., 0.47855339],
                 [0.47855339, 0.]]),
    ('TeGiGeT', [[0, 0],
                 [0, 0]]),
    ('DaGiGeTlD', [[0, 0],
                   [0, 0]]),
    ('DaGeTeGaD', [[0, 0],
                   [0, 0]]),
    ('TlDaGiGeT', [[0., 0.47855339],
                   [0., 0.]])
])
def test_dwpc_baab(metapath, expected):
    node_dict = {
        'G': ['CXCR4', 'IL2RA', 'IRF1', 'IRF8', 'ITCH', 'STAT3', 'SUMO1'],
        'D': ["Crohn's Disease", 'Multiple Sclerosis'],
        'T': ['Leukocyte', 'Lung']
    }
    exp_row = node_dict[metapath[0]]
    exp_col = node_dict[metapath[-1]]
    url = 'https://github.com/dhimmel/hetio/raw/{}/{}'.format(
        '9dc747b8fc4e23ef3437829ffde4d047f2e1bdde',
        'test/data/disease-gene-example-graph.json',
    )
    graph = hetio.readwrite.read_graph(url)
    metapath = graph.metagraph.metapath_from_abbrev(metapath)

    row, col, dwpc_matrix = dwpc_baab(graph, metapath, damping=0.5)

    expected = numpy.array(expected, dtype=numpy.float64)

    assert abs(dwpc_matrix - expected).sum() == pytest.approx(0, abs=1e-7)
    assert exp_row == row
    assert exp_col == col
