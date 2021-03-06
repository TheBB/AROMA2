from collections import namedtuple

import numpy as np
from scipy.linalg import eigh


# class Affinizer:


class PODBasis:

    def __init__(self, ensemble, norm, source=None):
        self.ensemble = ensemble
        self.norm = norm
        self.source = source
        self._eigvals = None
        self._eigvecs = None

    def suggest_source(self, name):
        if self.source is None:
            self.source = name

    def eigendata(self, case, mu, ndofs):
        if self._eigvals is not None and len(self._eigvals) >= ndofs:
            return self._eigvals[:ndofs], self._eigvecs[:,:ndofs]
        src = self.source

        mass = case.functions[self.norm](case, mu)[src, src]
        data = self.ensemble[src]
        nsnaps = len(data)

        corr = data.dot(mass.dot(data.T))
        eigval_range = (nsnaps - ndofs, nsnaps - 1)
        eigvals, eigvecs = eigh(corr, turbo=False, eigvals=eigval_range)
        del corr

        self._eigvals = eigvals[::-1]
        self._eigvecs = eigvecs[:,::-1]
        return self._eigvals[:ndofs], self._eigvecs[:,:ndofs]

    def projection(self, case, mu, ndofs):
        eigvals, eigvecs = self.eigendata(case, mu, ndofs)
        data = self.ensemble[self.source]
        return data.T.dot(eigvecs / np.sqrt(eigvals))


# Helper types for reduction
ProjBasis = namedtuple('ProjBasis', ['name'])
Contractable = namedtuple('Contractable', ['name'])


class Reducer:

    def __init__(self, case, mu=None):
        if mu is None:
            mu = case.parameter()
        self.case = case
        self.mu = mu
        self.bases = dict()
        self.liftspec = dict()

    def lift(self, name, lifts):
        self.liftspec[name] = lifts

    def _get_liftspec(self, name):
        if name in self.liftspec:
            return self.liftspec[name]
        return tuple(range(1, self.case.functions[name].ndim))

    def _derived_bases(self, src):
        return (name for name, basis in self.bases.items() if basis.source == src)

    def __setitem__(self, name, value):
        value.suggest_source(name)
        self.bases[name] = value

    def __call__(self, *args, **kwargs):
        ndofs = kwargs
        if len(args) > 0:
            for name in self.bases:
                ndofs.setdefault(name, args[0])
        projections = {
            name: basis.projection(self.case, self.mu, ndofs[name])
            for name, basis in self.bases.items()
        }

        for name, blocks in self.case.functions.items():
            for block, func in blocks.items():
                contracts = []
                for basisname in block:
                    axis_contracts = list(map(ProjBasis, self._derived_bases(basisname)))
                    axis_contracts.extend([
                        Contractable(cname)
                        for cname, cfunc in self.case.contractables.items()
                        if (basisname,) in cfunc
                    ])
                    contracts.append(axis_contracts)

                print(name, block, contracts)
