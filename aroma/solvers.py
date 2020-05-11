

# def _stokes_matrix(case, mu, div=True, **kwargs):
#     matrix = case['laplacian'](mu)
#     if div:
#         matrix += case['divergence'](mu, sym=True)
#     if 'stab-lhs' in case:
#         matrix += case['stab-lhs'](mu, sym=True)
#     return matrix


# def _stokes_rhs(case, mu, **kwargs):
#     rhs = - case['divergence'](mu, cont=('lift', None)) - case['laplacian'](mu, cont=(None, 'lift'))
#     if 'forcing' in case:
#         rhs += case['forcing'](mu)
#     if 'stab-lhs' in case:
#         rhs -= case['stab-lhs'](mu, cont=(None, 'lift'))
#     if 'stab-rhs' in case:
#         rhs += case['stab-rhs'](mu)
#     return rhs


# def _stokes_assemble(case, mu, **kwargs):
#     return _stokes_matrix(case, mu, **kwargs), _stokes_rhs(case, mu, **kwargs)


# def stokes(case, mu):
#     assert 'divergence' in case
#     assert 'laplacian' in case

#     matrix, rhs = _stokes_assemble(case, mu)
#     lhs = solve(matrix, rhs, case.constraints)

#     return lhs


def stokes(case, mu):
    assert 'divergence' in case
    assert 'laplacian' in case
