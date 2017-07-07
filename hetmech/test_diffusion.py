import numpy
import pytest

from .diffusion import diffusion_step


class TestDualNormalize:
    """
    Test hetmech.diffusion.dual_normalize()
    """

    def get_clean_matrix(self, dtype='float64'):
        """Return a newly allocated matrix."""
        matrix = [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ]
        matrix = numpy.array(matrix, dtype=dtype)
        return matrix

    @pytest.mark.parametrize('dtype', ['bool_', 'int8', 'float64'])
    def test_diffusion_step_passthrough(self, dtype):
        """Should not change matrix"""
        matrix = self.get_clean_matrix(dtype)
        output = diffusion_step(matrix, 0.0, 0.0)
        assert numpy.array_equal(output, matrix)

    @pytest.mark.parametrize('exponent', [0, 0.3, 0.5, 1, 2, 20])
    def test_diffusion_step_row_or_column_damping(self, exponent):
        """Test row, column damping individually"""
        # Create the matrix expected for single normalization
        p = exponent  # for easier reading
        expect = [
            [1/3**p, 1/3**p, 1/3**p],
            [1/2**p, 1/2**p, 0],
            [1, 0, 0],
        ]
        expect = numpy.array(expect, dtype='float64')

        # Test row normalization works as expected
        input_matrix = self.get_clean_matrix()
        matrix = diffusion_step(input_matrix, row_damping=exponent)
        assert numpy.allclose(expect, matrix)

        # Test column normalization works as expected
        input_matrix = self.get_clean_matrix()
        matrix = diffusion_step(input_matrix, column_damping=exponent)
        assert numpy.allclose(numpy.transpose(expect), matrix)

    @pytest.mark.parametrize('row_damping', [0, 0.3, 0.5, 1, 2])
    @pytest.mark.parametrize('column_damping', [0, 0.3, 0.5, 1, 2])
    def test_diffusion_step_row_and_column_damping(
            self, row_damping, column_damping):
        """Test simultaneous row and column damping"""
        input_matrix = self.get_clean_matrix()

        # Create the matrix expected for simultaneous dual normalization
        pr, pc = row_damping, column_damping
        expect = [
            [(1/3**pc) / (1/3**pc + 1/2**pc + 1)**pr,
             (1/2**pc) / (1/3**pc + 1/2**pc + 1)**pr,
             1 / (1/3**pc + 1/2**pc + 1)**pr],
            [(1/3**pc) / (1/3**pc + 1/2**pc)**pr,
             (1/2**pc) / (1/3**pc + 1/2**pc)**pr,
             0],
            [(1/3**pc) / (1/3**pc)**pr, 0, 0],
        ]
        expect = numpy.array(expect, dtype='float64')
        matrix = diffusion_step(input_matrix, row_damping, column_damping)
        assert numpy.allclose(expect, matrix)

    @staticmethod
    def get_problem_matrix(type):
        """Return a problematic matrix of specified type"""
        matrix_dict = {
            'empty_row': numpy.array([[1, 2], [3, 4], []]),
            'empty_matrix': numpy.array([[], [], []]),
            'nan_matrix': numpy.array([[numpy.nan, numpy.nan], [1, 0.5]]),
            'infinite_matrix': numpy.array([[numpy.inf, numpy.inf], [1, 0.5]]),
            'zero_matrix': numpy.zeros((3, 3))
        }
        return matrix_dict[type]

    @pytest.mark.parametrize('corner_type', ['empty_row', 'empty_matrix',
                                             'nan_matrix', 'infinite_matrix',
                                             'zero_matrix'])
    def test_corner_cases(self, corner_type):
        """
        Test that errors are returned for matrices with errors like NaN, Inf,
        emptiness, etc.
        """
        input_matrix = self.get_problem_matrix(corner_type)
        if corner_type == 'zero_matrix':  # Assert output is all zero
            assert not diffusion_step(input_matrix, 0.5, 0.5).any()
        else:
            with pytest.raises(AssertionError):
                assert diffusion_step(input_matrix, 0.5, 0.5)
