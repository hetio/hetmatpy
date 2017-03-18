import numpy

from .diffusion import dual_normalize


def test_dual_normalize():
    """
    First attempt at a pytest test.
    The test itself runs properly, and checks
    that dual_normalize is doing what it is supposed to.
    """
    toy_matrix = numpy.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0]
    ])
    toy_matrix = toy_matrix.astype('float64')
    toy_copy = toy_matrix.copy()
    test1 = dual_normalize(toy_copy, 0.0, 0.0)
    test1 = test1.astype('float64')
    assert(  numpy.sum(numpy.absolute(toy_matrix-test1)) == 0 )

    test_exponent = 0.5
    toy_copy = toy_matrix.copy()
    test2 = dual_normalize( toy_copy, test_exponent, 0.0 )
    test2.astype('float64')
    true_matrix2 = numpy.array([ 
        [1/3**test_exponent, 1/3**test_exponent, 1/3**test_exponent],
        [1/2**test_exponent, 1/2**test_exponent, 0],
        [1, 0, 0]
        ])
    assert(  numpy.sum(numpy.absolute(true_matrix2-test2)) < 10*numpy.finfo(float).eps )

    test_exponent = 0.3
    toy_copy = toy_matrix.copy()
    test3 = dual_normalize( toy_copy, 0, test_exponent )
    true_matrix3 = numpy.array([ 
        [1/3**test_exponent, 1/3**test_exponent, 1/3**test_exponent],
        [1/2**test_exponent, 1/2**test_exponent, 0],
        [1, 0, 0]
        ])
    true_matrix3 = numpy.transpose(true_matrix3)
    assert(  numpy.sum(numpy.absolute(true_matrix3-test3)) < 10*numpy.finfo(float).eps )
