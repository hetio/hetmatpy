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
            [(1/3**pr) / (1/3**pr + 1/2**pr + 1)**pc,
             (1/3**pr) / (1/3**pr + 1/2**pr)**pc,
             (1/3**pr) / (1/3**pr)**pc],
            [(1/2**pr) / (1/3**pr + 1/2**pr + 1)**pc,
             (1/2**pr) / (1/3**pr + 1/2**pr)**pc,
             0],
            [1 / (1/3**pr + 1/2**pr + 1)**pc, 0, 0],
        ]
        expect = numpy.array(expect, dtype='float64')
        matrix = diffusion_step(input_matrix, row_damping, column_damping)
        assert numpy.allclose(expect, matrix)

    @pytest.mark.parametrize('dtype', ['bool_', 'int8', 'float32', 'float64'])
    @pytest.mark.parametrize('copy', [True, False])
    def test_diffusion_step_copy(self, dtype, copy):
        """Test the copy argument of dual_normalize"""
        original = self.get_clean_matrix(dtype)
        input_matrix = original.copy()
        matrix = diffusion_step(input_matrix, 0.6, 0.9, copy=copy)
        # Test whether the original matrix is unmodified
        if copy or dtype != 'float64':
            assert numpy.array_equal(original, input_matrix)
            assert input_matrix is not matrix
        else:
            assert input_matrix is matrix
