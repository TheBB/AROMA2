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
    # assert 'system' in case

    mx = case.functions['system'](case, mu)
    mx += mx[~R['p'], 'p'].T

    rhs = FlexArray(ndim=1)
    rhs -= case.functions['system'](case, mu, block=('v','v'), contract=(None, lift))
    rhs -= case.functions['system'](case, mu, block=('v','p'), contract=(lift, None))

    return solve(mx, rhs, case.cons, R['v','p'])


def supremizer(mu, rhs, case):
    # assert 'system' in case
    # assert 'v-h1s' in case

    mx = case.functions['v-h1s'](case, mu)
    rhs = case.functions['system'](case, mu, block=('v','p'), contract=(None, rhs))
    return solve(mx, rhs, case.cons, R['v'])
