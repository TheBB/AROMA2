from nutils import matrix, function as fn

from aroma.affine import ParameterDependent, Basis
from aroma.case import HifiCase
from aroma.util import apply_contraction, dependency_union, COOSparse


class SparseBackend(matrix.Backend):

    def __init__(self, shape, indices):
        self.shape = shape
        self.master = indices

    def assemble(self, data, index, shape):
        index = tuple(i if m is None else m[i] for m, i in zip(self.master, index))
        return COOSparse((data, index), shape=self.shape)


class NutilsBasis(Basis):
    """Basis class for holding a Nutils function object."""

    basis: object

    def __init__(self, basis):
        super().__init__(shape=(basis.shape))
        self.basis = basis

    def evaluate(self, case, mu, contract, **kwargs):
        return apply_contraction(self.basis, contract)


class NutilsCase(HifiCase):
    """Case class specialized for using Nutils as a backend."""

    domain: object

    def __init__(self, name, domain):
        super().__init__(name)
        self.domain = domain

    def sparse_integrate(self, itg, preshape):
        shape = tuple(self.ndofs if isinstance(s, str) else s for s in preshape)
        indices = tuple(self.bases[s].indices if isinstance(s, str) else None for s in preshape)
        with SparseBackend(shape, indices):
            return self.domain.integrate(itg, ischeme='gauss9')


class Laplacian(ParameterDependent):
    """Nutils Laplacian matrix."""

    basisname: str

    def __init__(self, case, basisname, **kwargs):
        basis = case.bases[basisname]
        shape = (len(basis),) * 2
        dependencies = dependency_union(basis, case.geometry_func)
        super().__init__(shape, dependencies, **kwargs)
        self.basisname = basisname

    def evaluate(self, case, mu, contract, **kwargs):
        geom = case.geometry(mu)
        basis = case.basis(self.basisname, mu)

        itg = fn.outer(basis.grad(geom))
        sum_dims = list(range(2, itg.ndim))
        itg = apply_contraction(itg.sum(sum_dims), contract)
        return case.sparse_integrate(itg * fn.J(geom), (self.basisname,) * 2)


class Divergence(ParameterDependent):
    """Nutils divergence matrix."""

    vbasisname: str
    pbasisname: str

    def __init__(self, case, vbasisname, pbasisname, **kwargs):
        vbasis = case.bases[vbasisname]
        pbasis = case.bases[pbasisname]
        shape = (len(vbasis), len(pbasis))
        dependencies = dependency_union(vbasis, pbasis, case.geometry_func)
        super().__init__(shape, dependencies, **kwargs)
        self.vbasisname = vbasisname
        self.pbasisname = pbasisname

    def evaluate(self, case, mu, contract, **kwargs):
        geom = case.geometry(mu)
        vbasis = case.basis(self.vbasisname, mu)
        pbasis = case.basis(self.pbasisname, mu)

        itg = -fn.outer(vbasis.div(geom), pbasis)
        itg = apply_contraction(itg, contract)
        return case.sparse_integrate(itg * fn.J(geom), (self.vbasisname, self.pbasisname))
