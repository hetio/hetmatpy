from hetmatpy.pipeline import (
    grouper,
    calculate_sd,
    calculate_p_value
)
import pytest


def test_grouper_equal_chunks():
    iterable = range(10)
    grouped = grouper(iterable, group_size=2)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]


def test_grouper_ragged_chunks():
    iterable = range(7)
    grouped = grouper(iterable, group_size=3)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0, 1, 2),
        (3, 4, 5),
        (6,),
    ]


def test_grouper_one_group():
    iterable = range(7)
    grouped = grouper(iterable, group_size=20)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0, 1, 2, 3, 4, 5, 6),
    ]


def test_grouper_length_1():
    iterable = range(4)
    grouped = grouper(iterable, group_size=1)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]


@pytest.mark.parametrize('sum_of_squares, unsquared_sum, number_nonzero, expected_output', [
    # Sum of squares and unsquared sum are from a pair of the same number, so return zero
    (32, 8, 2, 0),
    # Sum of squares and unsquared sum are very close to the case above, so return zero
    (32, 8 + 1e-6, 2, 0),
    # Only one nonzero observation, so return zero
    (5, 5, 1, 0),
    # Test that the standard deviation of 5, 4, and 3 is 1
    (50, 12, 3, 1)])
def test_calculate_sd(sum_of_squares, unsquared_sum, number_nonzero, expected_output):
    assert calculate_sd(sum_of_squares, unsquared_sum, number_nonzero) == expected_output


# row = path_count, sd_nz, dwpc, nnz, n, alpha, beta
# dgp_df = mean_nz, nnz, n_dwpcs
@pytest.mark.parametrize('row, dgp_df, expected_output', [
    # Zero path count
    ({'path_count': 0, 'sd_nz': 2, 'dwpc': 4, 'nnz': 1, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 1},
     {'mean_nz': 3, 'nnz': 1, 'n_dwpcs': 1}, 1),
    # zero standard deviation with dwpc lower than mean
    ({'path_count': 5, 'sd_nz': 0, 'dwpc': 2, 'nnz': 1, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 1},
     {'mean_nz': 3, 'nnz': 3, 'n_dwpcs': 8}, .375),
    # zero standard deviation with dwpc higher than mean
    ({'path_count': 5, 'sd_nz': 0, 'dwpc': 4, 'nnz': 1, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 1},
     {'mean_nz': 3, 'nnz': 3, 'n_dwpcs': 8}, 0),
    # Normal gamma hurdle case
    ({'path_count': 5, 'sd_nz': 1, 'dwpc': 2.5, 'nnz': 1, 'n': 10, 'alpha': 1, 'beta': 1, 'sum': 1},
     {'mean_nz': 3, 'nnz': 3, 'n_dwpcs': 8}, .008208),
    # number nonzero is itself zero
    ({'path_count': 5, 'sd_nz': 0, 'dwpc': 2, 'nnz': 0, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 0},
     {'mean_nz': 3, 'nnz': 3, 'n_dwpcs': 8}, 0)])
def test_calculate_p_value(row, dgp_df, expected_output):
    assert calculate_p_value(row, dgp_df) == pytest.approx(expected_output, rel=1e-4)
