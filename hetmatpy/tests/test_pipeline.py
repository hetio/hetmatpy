from hetmatpy.pipeline import (
    grouper,
    calculate_sd,
    calculate_p_value,
    add_gamma_hurdle_to_dgp_df
)
import numpy
import pandas
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
    (5, 5, 1, None),
    # Test that the standard deviation of 5, 4, and 3 is 1
    (50, 12, 3, 1)])
def test_calculate_sd(sum_of_squares, unsquared_sum, number_nonzero, expected_output):
    assert calculate_sd(sum_of_squares, unsquared_sum, number_nonzero) == expected_output


@pytest.mark.parametrize('row, expected_output', [
    # Zero path count
    ({'path_count': 0, 'sd_nz': 2, 'dwpc': 4, 'nnz': 1, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 1}, 1),
    # zero standard deviation with dwpc lower than mean
    ({'path_count': 5, 'sd_nz': 0, 'dwpc': 2, 'mean_nz': 3, 'nnz': 3, 'n_dwpcs': 8, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 1}, .375),
    # zero standard deviation with dwpc higher than mean
    ({'path_count': 5, 'sd_nz': 0, 'dwpc': 4, 'mean_nz': 3, 'nnz': 1, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 1}, 0),
    # Normal gamma hurdle case
    ({'path_count': 5, 'sd_nz': 1, 'dwpc': 2.5, 'nnz': 1, 'n': 10, 'alpha': 1, 'beta': 1, 'sum': 1}, .008208),
    # number nonzero is itself zero
    ({'path_count': 5, 'sd_nz': 0, 'dwpc': 2, 'nnz': 0, 'n': 4, 'alpha': 1, 'beta': 2, 'sum': 0}, 0)])
def test_calculate_p_value(row, expected_output):
    assert calculate_p_value(row) == pytest.approx(expected_output, rel=1e-4)


def test_add_gamma_hurdle():
    df_dict = {'nnz': [1, 3, 3],
               'sum': [4, 4, 3],
               'sum_of_squares': [4, 6, 3 + 1e-15]}
    dgp_df = pandas.DataFrame(df_dict)
    dgp_df = add_gamma_hurdle_to_dgp_df(dgp_df)

    # Test nnz = 1
    expected_mean_nz_0 = 4
    assert expected_mean_nz_0 == dgp_df['mean_nz'][0]
    assert numpy.isnan(dgp_df['sd_nz'][0])
    assert numpy.isnan(dgp_df['beta'][0])
    assert numpy.isnan(dgp_df['alpha'][0])

    # Test a normal case
    expected_mean_nz_1 = 4 / 3
    expected_sd_nz_1 = ((2 / 3) / 2) ** .5
    expected_beta_1 = expected_mean_nz_1 / expected_sd_nz_1 ** 2
    expected_alpha_1 = expected_mean_nz_1 * expected_beta_1
    assert expected_mean_nz_1 == dgp_df['mean_nz'][1]
    assert expected_sd_nz_1 == pytest.approx(dgp_df['sd_nz'][1])
    assert expected_beta_1 == pytest.approx(dgp_df['beta'][1])
    assert expected_alpha_1 == pytest.approx(dgp_df['alpha'][1])

    # Test squared deviations ~ 0
    expected_mean_nz_2 = 1
    expected_sd_nz_2 = 0
    assert expected_mean_nz_2 == dgp_df['mean_nz'][2]
    assert expected_sd_nz_2 == dgp_df['sd_nz'][2]
    assert numpy.isnan(dgp_df['beta'][2])
    assert numpy.isnan(dgp_df['alpha'][2])
