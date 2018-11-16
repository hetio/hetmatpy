import pytest

import hetmech.degree_weight
import hetmech.hetmat.caching
from hetmech.tests.hetnets import get_graph


@pytest.mark.parametrize('allocate_GB', [0, 0.1])
def test_path_count_priority_cache(tmpdir, allocate_GB):
    """
    Test PathCountPriorityCache by runnin the same DWWC computation three times.
    """
    hetmat = get_graph('bupropion-subgraph', hetmat=True, directory=tmpdir)
    cache = hetmech.hetmat.caching.PathCountPriorityCache(hetmat, allocate_GB)
    hetmat.path_counts_cache = cache
    print(cache.get_stats)

    # First run
    assert sum(cache.hits.values()) == 0
    row_ids, col_ids, matrix = hetmech.degree_weight.dwwc(
        graph=hetmat, metapath='CbGpPWpGaD', damping=0.5,
        dwwc_method=hetmech.degree_weight.dwwc_recursive,
    )
    assert sum(cache.hits.values()) > 0
    if allocate_GB == 0:
        assert cache.hits['memory'] == 0
        assert cache.hits['disk'] == 0
        assert cache.hits['absent'] == 4
    elif allocate_GB > 0:
        assert cache.hits['memory'] == 0
        assert cache.hits['disk'] == 0
        assert cache.hits['absent'] == 4

    # Second run
    row_ids, col_ids, matrix = hetmech.degree_weight.dwwc(
        graph=hetmat, metapath='CbGpPWpGaD', damping=0.5,
        dwwc_method=hetmech.degree_weight.dwwc_recursive,
    )
    if allocate_GB == 0:
        assert cache.hits['memory'] == 0
        assert cache.hits['disk'] == 0
        assert cache.hits['absent'] == 8
    elif allocate_GB > 0:
        assert cache.hits['memory'] == 1
        assert cache.hits['disk'] == 0
        assert cache.hits['absent'] == 4

    # Save DWWC matrix
    path = hetmat.get_path_counts_path('CbGpPWpGaD', 'dwwc', 0.5, 'npy')
    path.parent.mkdir(parents=True)
    hetmech.hetmat.save_matrix(matrix, path)

    # Third run
    row_ids, col_ids, matrix = hetmech.degree_weight.dwwc(
        graph=hetmat, metapath='CbGpPWpGaD', damping=0.5,
        dwwc_method=hetmech.degree_weight.dwwc_recursive,
    )
    if allocate_GB == 0:
        assert cache.hits['memory'] == 0
        assert cache.hits['disk'] == 1
        assert cache.hits['absent'] == 8
    elif allocate_GB > 0:
        assert cache.hits['memory'] == 2
        assert cache.hits['disk'] == 0
        assert cache.hits['absent'] == 4
    print(cache.get_stats)
