from typing import Tuple

import numpy as np
from nutils import matrix, function as fn

from aroma.affine import ParameterDependent, Basis
from aroma.case import HifiCase
from aroma.util import apply_contraction, dependency_union, NamedBlock


class NutilsBasis(Basis):
    """Basis class for holding a Nutils function object."""

    basis: object

    def __init__(self, basis):
        super().__init__(shape=(basis.shape))
        self.basis = basis

    def evaluate(self, case, mu, contract, **kwargs):
        obj, _ = apply_contraction(self.basis, (self.name,), contract)
        return obj


class NutilsCase(HifiCase):
    """Case class specialized for using Nutils as a backend."""

    domain: object

    def __init__(self, name, domain):
        super().__init__(name)
        self.domain = domain

    def sparse_integrate(self, itg, basisnames):
        with matrix.Scipy():
            retval = self.domain.integrate(itg, ischeme='gauss9')
        if isinstance(retval, matrix.Matrix):
            retval = retval.core
        return NamedBlock(basisnames, retval)

    def project_function(self, func, basisname, mu=None):
        if mu is None:
            mu = self.parameter()
        geom = self.geometry(mu)
        basis = self.basis(basisname, mu)
        with matrix.Scipy():
            lift = self.domain.project(func, onto=basis, geometry=geom, ischeme='gauss9')
        return NamedBlock(basisname, lift)

    def constrain(self, basisname, *boundaries, component=None, mu=None):
        if len(boundaries) == 1 and isinstance(boundaries[0], np.ndarray):
            return super().constrain(basisname, *boundaries)

        if mu is None:
            mu = self.parameter()
        boundary = self.domain.boundary[','.join(boundaries)]

        geom = self.geometry(mu)
        basis = self.basis(basisname, mu)
        zero = np.zeros(basis.shape[1:])
        if component is not None:
            basis = basis[...,component]
            zero = zero[...,component]

        with matrix.Scipy():
            projected = boundary.project(zero, onto=basis, geometry=geom, ischeme='gauss2')
        super().constrain(basisname, projected)


class NutilsIntegrand(ParameterDependent):
    """Superclass for Nutils integrand objects."""

    basisnames: Tuple[str, ...]

    def __init__(self, case, basisnames, **kwargs):
        bases = [case.bases[name] for name in basisnames]
        shape = tuple(len(basis) for basis in bases)
        dependencies = dependency_union(case.geometry_func, *bases)
        super().__init__(shape, dependencies, **kwargs)
        self.basisnames = basisnames

    def contract_and_integrate(self, case, itg, contract, geom):
        itg, names = apply_contraction(itg, self.basisnames, contract)
        return case.sparse_integrate(itg * fn.J(geom), names)


class Laplacian(NutilsIntegrand):
    """Nutils Laplacian matrix."""

    def __init__(self, case, basisname, **kwargs):
        super().__init__(case, (basisname,) * 2, **kwargs)

    def evaluate(self, case, mu, contract, **kwargs):
        geom = case.geometry(mu)
        basis = case.basis(self.basisnames[0], mu)
        itg = fn.outer(basis.grad(geom))
        sum_dims = list(range(2, itg.ndim))
        return self.contract_and_integrate(case, itg.sum(sum_dims), contract, geom)


class Divergence(NutilsIntegrand):
    """Nutils divergence matrix."""

    def __init__(self, case, vbasisname, pbasisname, **kwargs):
        super().__init__(case, (vbasisname, pbasisname), **kwargs)

    def evaluate(self, case, mu, contract, **kwargs):
        geom = case.geometry(mu)
        vbasis = case.basis(self.basisnames[0], mu)
        pbasis = case.basis(self.basisnames[1], mu)
        itg = -fn.outer(vbasis.div(geom), pbasis)
        return self.contract_and_integrate(case, itg, contract, geom)
