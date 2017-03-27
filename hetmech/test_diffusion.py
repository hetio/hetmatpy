import numpy
import pytest

from .diffusion import dual_normalize


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
    def test_dual_normalize_passthrough(self, dtype):
        """Should not change matrix"""
        matrix = self.get_clean_matrix(dtype)
        output = dual_normalize(matrix, 0.0, 0.0)
        assert numpy.array_equal(output, matrix)

    @pytest.mark.parametrize('exponent', [0, 0.3, 0.5, 1, 2, 20])
    @pytest.mark.parametrize('dtype', ['bool_', 'int8', 'float32', 'float64'])
    @pytest.mark.parametrize('copy', [True, False])
    def test_dual_normalize_row_or_column_damping(self, exponent, dtype, copy):
        """Test row, column damping individually"""
        original = self.get_clean_matrix(dtype)

        # Create the matrix expected for single normalization
        p = exponent  # for easier reading
        expect = [
            [1/3**p, 1/3**p, 1/3**p],
            [1/2**p, 1/2**p, 0],
            [1, 0, 0],
        ]
        expect = numpy.array(expect, dtype='float64')

        # Test row normalization works as expected
        input_matrix = original.copy()
        matrix = dual_normalize(input_matrix, p, 0.0, copy=copy)
        assert numpy.allclose(expect, matrix)

        # Test column normalization works as expected
        input_matrix = original.copy()
        matrix = dual_normalize(input_matrix, 0.0, p, copy=copy)
        assert numpy.allclose(numpy.transpose(expect), matrix)

        # Test whether the original matrix is unmodified
        if copy or dtype != 'float64':
            assert numpy.array_equal(original, input_matrix)
        else:
            assert input_matrix is matrix

    @pytest.mark.parametrize('exponents',
                             [[0, 0], [0, 0.3], [0.3, 0], [0.5, 1], [1, 0.5]])
    @pytest.mark.parametrize('dtype', ['bool_', 'int8', 'float32', 'float64'])
    @pytest.mark.parametrize('copy', [True, False])
    def test_dual_normalize_row_and_column_damping(self,
                                                   exponents, dtype, copy):
        """Test simultaneous row and column damping"""
        original = self.get_clean_matrix(dtype)

        # Create the matrix expected for simultaneous dual normalization
        pr, pc = exponents
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
        input_matrix = original.copy()
        matrix = dual_normalize(input_matrix, pr, pc, copy=copy)
        assert numpy.allclose(expect, matrix)

        # Test whether the original matrix is unmodified
        if copy or dtype != 'float64':
            assert numpy.array_equal(original, input_matrix)
        else:
            assert input_matrix is matrix
