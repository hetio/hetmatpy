import numpy
import pandas
import pytest

from hetmatpy.pipeline import (
    grouper,
    calculate_sd,
    calculate_p_value,
    add_gamma_hurdle_to_dgp_df,
    path_does_not_exist,
)


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
    (32.0, 8.0, 2, 0.0),
    # Sum of squares and unsquared sum are very close to the case above, so return zero
    (32.0, 8.0 + 1e-6, 2, 0.0),
    # Only one nonzero observation, so return None
    (5.0, 5.0, 1, None),
    # Test that the standard deviation of 5, 4, and 3 is 1
    (50.0, 12.0, 3, 1.0),
    # Test no nonzero values
    (0.0, 0.0, 0, None),
])
def test_calculate_sd(sum_of_squares, unsquared_sum, number_nonzero, expected_output):
    assert calculate_sd(sum_of_squares, unsquared_sum, number_nonzero) == expected_output


@pytest.mark.parametrize('row, expected_output', [
    # zero path count
    ({'path_count': 0,
      'sd_nz': 2.0,
      'dwpc': 4.0,
      'nnz': 1,
      'n': 4,
      'alpha': 1.0,
      'beta': 2.0,
      'sum': 1.0
      }, 1.0),
    # zero standard deviation with dwpc lower than mean
    ({'path_count': 5,
      'sd_nz': 0.0,
      'dwpc': 2.0,
      'mean_nz': 3.0,
      'nnz': 3,
      'n': 8,
      'alpha': 1.0,
      'beta': 2.0,
      'sum': 1.0
      }, .375),
    # zero standard deviation with dwpc higher than mean
    ({'path_count': 5,
      'sd_nz': 0.0,
      'dwpc': 4.0,
      'mean_nz': 3.0,
      'nnz': 1,
      'n': 4,
      'alpha': 1.0,
      'beta': 2.0,
      'sum': 1.0
      }, 0.0),
    # normal gamma hurdle case
    ({'path_count': 5,
      'sd_nz': 1.0,
      'dwpc': 2.5,
      'nnz': 1,
      'n': 10,
      'alpha': 1.0,
      'beta': 1.0,
      'sum': 1.0
      }, .008208),
    # number nonzero is itself zero
    ({'path_count': 5,
      'sd_nz': 0.0,
      'dwpc': 2.0,
      'nnz': 0,
      'n': 4,
      'alpha': 1.0,
      'beta': 2.0,
      'sum': 0.0
      }, 0.0),
    # dwpc slightly larger than mean_nz, but within float error tolerance
    ({'source_id': 'DB00193',
      'target_id': 'DOID:0050425',
      'source_name': 'Tramadol',
      'target_name': 'restless legs syndrome',
      'source_degree': 1,
      'target_degree': 10,
      'path_count': 1,
      'dwpc': 7.323728709931218,
      'n': 81600,
      'nnz': 2086,
      'n_perms': 200,
      'mean_nz': 7.323728709931212,
      'sd_nz': 0.0
      }, 0.02556372549),
])
def test_calculate_p_value(row, expected_output):
    assert calculate_p_value(row) == pytest.approx(expected_output, rel=1e-4)


def test_add_gamma_hurdle():
    df_dict = {'nnz': [1, 3, 3],
               'sum': [4.0, 4.0, 3.0],
               'sum_of_squares': [4.0, 6.0, 3.0 + 1e-15],
               }
    dgp_df = pandas.DataFrame(df_dict)
    dgp_df = add_gamma_hurdle_to_dgp_df(dgp_df)

    # Test nnz = 1
    expected_mean_nz_0 = 4.0
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
    expected_mean_nz_2 = 1.0
    expected_sd_nz_2 = 0.0
    assert expected_mean_nz_2 == dgp_df['mean_nz'][2]
    assert expected_sd_nz_2 == dgp_df['sd_nz'][2]
    assert numpy.isnan(dgp_df['beta'][2])
    assert numpy.isnan(dgp_df['alpha'][2])


@pytest.mark.parametrize('row, expected_output', [
    # Path count is zero
    ({'path_count': 0.0}, True),
    # Path count is nonzero
    ({'path_count': 1.0}, False),
    # No path count, dwpc is zero
    ({'dwpc': 0.0}, True),
    # No path count, dwpc is nonzero
    ({'dwpc': .01}, False),
])
def test_path_does_not_exist(row, expected_output):
    assert path_does_not_exist(row) == expected_output
