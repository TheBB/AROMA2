import scipy
import scipy.sparse as sparse
import numpy as np
from nutils import matrix


def solve(case, mx, rhs, solver='direct', **kwargs):
    mx = mx.realize()
    rhs = rhs.realize()

    if solver == 'mkl':
        if isinstance(mx, np.ndarray):
            raise TypeError
        mx = sparse.coo_matrix(mx)
        mx = matrix.MKLMatrix(mx.data, np.array([mx.row, mx.col]), mx.shape)
        return mx.solve(rhs, constrain=case.cons, **kwargs)

    elif isinstance(mx, np.matrix):
        mx = matrix.NumpyMatrix(np.array(mx))
        return mx.solve(rhs, constrain=case.cons, **kwargs)

    elif isinstance(mx, np.ndarray):
        mx = matrix.NumpyMatrix(mx)
        return mx.solve(rhs, constrain=case.cons, **kwargs)

    else:
        mx = matrix.ScipyMatrix(mx, scipy)
        return mx.solve(rhs, constrain=case.cons, solver=solver, **kwargs)


def lhs_wrap(case, lhs, *bases):
    retval = case.block_assembler(bases)
    for basisname in bases:
        retval[basisname] = lhs[case.bases[basisname].indices]
    return retval


def stokes(case, mu, vlift='lift/v'):
    assert 'laplacian' in case
    assert 'divergence' in case

    mx = case.block_assembler(['v', 'p'], repeat=2)
    mx += case['laplacian'](mu)
    divergence = case['divergence'](mu)
    mx += divergence
    mx += divergence.T

    rhs = case.block_assembler(['v', 'p'])
    rhs += case['laplacian'](mu, contract=(None, vlift))
    rhs += case['divergence'](mu, contract=(vlift, None))

    return lhs_wrap(case, solve(case, mx, rhs), 'v', 'p')
