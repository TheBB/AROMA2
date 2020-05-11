import scipy
import scipy.sparse as sparse
import numpy as np
from nutils import matrix


def solve(mx, rhs, cons, solver='direct', **kwargs):
    if solver == 'mkl':
        if isinstance(mx, np.ndarray):
            raise TypeError
        mx = sparse.coo_matrix(mx)
        mx = matrix.MKLMatrix(mx.data, np.array([mx.row, mx.col]), mx.shape)
        return mx.solve(rhs, constrain=cons, **kwargs)

    elif isinstance(mx, np.matrix):
        mx = matrix.NumpyMatrix(np.array(mx))
        return mx.solve(rhs, constrain=cons, **kwargs)

    elif isinstance(mx, np.ndarray):
        mx = matrix.NumpyMatrix(mx)
        return mx.solve(rhs, constrain=cons, **kwargs)

    else:
        mx = matrix.ScipyMatrix(mx, scipy)
        return mx.solve(rhs, constrain=cons, solver=solver, **kwargs)


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

    solve(mx.realize(), rhs.realize(), case.cons)
