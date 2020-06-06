from flexarrays import FlexArray, R
import scipy
import scipy.sparse as sparse
import numpy as np
from nutils import matrix


def solve(fmx, frhs, fcons, names, solver='direct', **kwargs):
    mx = fmx.realize[names, names]
    rhs = frhs.realize[names]
    cons = fcons.realize[names]

    if solver == 'mkl':
        if isinstance(mx, np.ndarray):
            raise TypeError
        mx = sparse.coo_matrix(mx)
        mx = matrix.MKLMatrix(mx.data, np.array([mx.row, mx.col]), mx.shape)
        retval = mx.solve(rhs, constrain=cons, **kwargs)

    elif isinstance(mx, np.matrix):
        mx = matrix.NumpyMatrix(np.array(mx))
        retval = mx.solve(rhs, constrain=cons, **kwargs)

    elif isinstance(mx, np.ndarray):
        mx = matrix.NumpyMatrix(mx)
        retval = mx.solve(rhs, constrain=cons, **kwargs)

    else:
        mx = matrix.ScipyMatrix(mx, scipy)
        retval = mx.solve(rhs, constrain=cons, solver=solver, **kwargs)

    return fmx.compatible(names, retval)


def stokes(mu, case, lift='lift'):
    assert 'laplacian' in case
    assert 'divergence' in case

    mx = FlexArray(ndim=2)
    mx += case['laplacian'](mu)
    divergence = case['divergence'](mu)
    mx += divergence
    mx += divergence.T

    rhs = FlexArray(ndim=1)
    rhs -= case['laplacian'](mu, contract=(None, lift))
    rhs -= case['divergence'](mu, contract=(lift, None))

    return solve(mx, rhs, case.cons, R['v','p'])


def supremizer(mu, rhs, case):
    assert 'laplacian' in case
    assert 'divergence' in case

    mx = case['v-h1s'](mu)
    rhs = case['divergence'](mu, contract=(None, rhs))
    return solve(mx, rhs, case.cons, R['v'])
