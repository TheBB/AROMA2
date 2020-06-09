from functools import partial
from typing import Tuple

from flexarrays import FlexArray
import flexarrays
import numpy as np
from nutils import matrix, function as fn

from aroma.affine import ParameterDependent, Basis
from aroma.case import HifiCase
from aroma.util import dependency_union


@flexarrays.copy.register(fn.Array)
def _(obj):
    return obj

@flexarrays.transpose.register(fn.Array)
def _(obj, perm):
    return obj.transpose(perm)

@flexarrays.contract.register(fn.Array, np.ndarray, int)
def _(a, b, axis):
    assert b.ndim == 1
    b = b.reshape(tuple(1 if i != axis else -1 for i in range(a.ndim)))
    retval = (a * b).sum(axis)
    return retval


class NutilsBasis(Basis):
    """Basis class for holding a Nutils function object."""

    basis: object

    def __init__(self, basis):
        super().__init__((len(basis),))
        self.basis = basis

    def __len__(self):
        return len(self.basis)

    def evaluate(self, case, mu, contract, **kwargs):
        c, *_ = contract
        if c is not None:
            return self.basis.dot(c[self.name])
        return self.basis


class NutilsCase(HifiCase):
    """Case class specialized for using Nutils as a backend."""

    domain: object
    geometry_func: ParameterDependent

    def __init__(self, name, domain):
        super().__init__(name)
        self.domain = domain

    @property
    def geometry(self):
        return partial(self.geometry_func, self)

    @geometry.setter
    def geometry(self, value):
        self.geometry_func = value

    @property
    def sampler(self):
        return self.domain.sample('bezier', degree=2)

    def triangulation(self):
        return self.sampler.tri

    def discretized_geometry(self, mu):
        geom = self.geometry(mu)
        return self.sampler.eval(geom)

    def discretized(self, mu, lhs, basisname, lift=True):
        if lift:
            lhs = lhs + self.contractables['lift'](self, mu)
        basis = self.basis(basisname, mu)
        solution = basis.dot(lhs[basisname])
        return self.sampler.eval(solution)

    def sparse_integrate(self, itg):
        with matrix.Scipy():
            retval = self.domain.integrate(itg, ischeme='gauss', degree=9)
        if isinstance(retval, matrix.Matrix):
            retval = retval.core
        return retval

    def project_function(self, func, basisname, mu=None):
        if mu is None:
            mu = self.parameter()
        geom = self.geometry(mu)
        basis = self.basis(basisname, mu)
        with matrix.Scipy():
            lift = self.domain.project(func, onto=basis, geometry=geom, ischeme='gauss', degree=9)
        return FlexArray.vector(basisname, lift)

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
            projected = boundary.project(zero, onto=basis, geometry=geom, ischeme='gauss', degree=2)
        super().constrain(basisname, projected)


class NutilsIntegrand(ParameterDependent):
    """Superclass for Nutils integrand objects."""

    def __init__(self, case, basisnames, **kwargs):
        bases = [case.bases[name] for name in basisnames]
        dependencies = dependency_union(case.geometry_func, *bases)
        shape = tuple(map(len, bases))
        super().__init__(shape, dependencies, **kwargs)


def contract_and_integrate(case, itg, contract, block):
    itg = FlexArray.single(block, itg)
    itg = itg.contract_many(contract)
    names, itg = itg.only()
    retval = case.sparse_integrate(itg)
    return FlexArray.single(names, retval)


class Laplacian(NutilsIntegrand):
    """Nutils Laplacian matrix."""

    def __init__(self, case, basisname, **kwargs):
        super().__init__(case, (basisname, basisname), **kwargs)

    def evaluate(self, case, mu, contract, block, **kwargs):
        geom = case.geometry(mu)
        bname, _ = block
        basis = case.basis(bname, mu)
        itg = fn.outer(basis.grad(geom))
        sum_dims = list(range(2, itg.ndim))
        return contract_and_integrate(case, itg.sum(sum_dims) * fn.J(geom), contract, block)

def laplacian(case, basisname, **kwargs):
    func = Laplacian(case, basisname, **kwargs)
    return FlexArray.single((basisname, basisname), func)


class Divergence(NutilsIntegrand):
    """Nutils divergence matrix."""

    def __init__(self, case, vname, pname, **kwargs):
        super().__init__(case, (vname, pname), **kwargs)

    def evaluate(self, case, mu, contract, block, **kwargs):
        geom = case.geometry(mu)
        vname, pname = block
        vbasis = case.basis(vname, mu)
        pbasis = case.basis(pname, mu)
        itg = -fn.outer(vbasis.div(geom), pbasis)
        return contract_and_integrate(case, itg * fn.J(geom), contract, block)

def divergence(case, vname, pname, **kwargs):
    func = Divergence(case, vname, pname, **kwargs)
    return FlexArray.single((vname, pname), func)


class Mass(NutilsIntegrand):
    """Nutils mass matrix."""

    def __init__(self, case, basisname, **kwargs):
        super().__init__(case, (basisname, basisname), **kwargs)

    def evaluate(self, case, mu, contract, block, **kwargs):
        geom = case.geometry(mu)
        bname, _ = block
        basis = case.basis(bname, mu)
        itg = fn.outer(basis)
        return contract_and_integrate(case, itg * fn.J(geom), contract, block)

def mass(case, basisname, **kwargs):
    func = Mass(case, basisname, **kwargs)
    return FlexArray.single((basisname, basisname), func)
