from __future__ import annotations
from typing import Dict, Tuple, Optional

from filebacked import FileBacked
import numpy as np

from aroma.util import broadcast_shapes, COOSparse
from aroma.mufunc import MuFunc


class ParameterDependent(FileBacked):
    """Ultimate superclass for any parameter-dependent function, affine
    representation or otherwise.

    Subclasses must implement the evaluate method.
    """

    shape: Tuple[int, ...]
    dependencies: Tuple[str, ...]
    explicit_scale: Optional[MuFunc]
    cached_contractions: Dict[Tuple[str, ...], ParameterDependent]

    def __init__(self, shape=(), dependencies=(), explicit_scale=None):
        super().__init__()
        self.shape = shape
        self.dependencies = dependencies
        self.explicit_scale = explicit_scale
        self.cached_contractions = dict()

    @property
    def ndim(self):
        return len(self.shape)

    def __call__(self, case, mu, contract=None, **kwargs):
        """Evaluate this function at the parametric point mu.

        The 'case' argument must be a reference to the relevant case
        object.

        If 'contract' is given, it must be a tuple of length equal to
        the number of dimensions.  Strings are resolved by evaluation
        according to 'contractables' into vectors.  Vectors are also
        supported.  A value of None indicates no contraction over this
        dimension.
        """

        if contract is None:
            contract = (None,) * self.ndim

        # Check if this contraction pattern has been cached
        contract_pattern = tuple(c if isinstance(c, str) else None for c in contract)
        if contract_pattern in self.cached_contractions:
            # Call the cached function.  It is likely to be much faster.
            sub_contract = tuple(c for c in contract if not isinstance(c, str))
            retval = self.cached_contractions[contract_pattern](
                case, mu, contract=sub_contract, **kwargs,
            )
        else:
            # Handle contractions manually by evaluating them to vectors
            contractables = getattr(case, 'contractables', None)
            if contractables:
                contract = tuple(
                    contractables[c](mu, contractables) if isinstance(c, str) else c
                    for c in contract
                )
            retval = self.evaluate(case, mu, contract, **kwargs)

        if isinstance(retval, COOSparse):
            retval = retval.csr_matrix()

        if self.explicit_scale:
            return retval * self.explicit_scale(mu)
        return retval

    def evaluate(self, case, mu, contract, **kwargs):
        """Evaluate this function at the parametric point mu.

        The 'contract' argument is a tuple of length equal to the
        number of dimensions, with elements optionally vector-valued.
        All vector-valued dimensions must be contracted in the final
        output.
        """
        raise NotImplementedError


class ParameterLambda(ParameterDependent):
    """Helper class for wrapping a function as a ParameterDependent
    object.

    The function must accept two arguments: a case object and a
    parametric point.
    """

    func: object

    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def evaluate(self, case, mu, contract, **kwargs):
        return self.func(case, mu, contract, **kwargs)


class ParameterContainer(ParameterDependent):
    """A dictionary-like container that wraps multiple named
    ParameterDependent objects as a sum.
    """

    data: Dict[str, ParameterDependent]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = dict()

    def recompute(self):
        self.shape = broadcast_shapes(self.values())
        deps = set()
        for sub in self.values():
            deps |= set(sub.dependencies)
        self.dependencies = deps

    def evaluate(self, *args, **kwargs):
        sub_iter = self.values()
        retval = next(sub_iter).evaluate(*args, **kwargs)
        for sub in sub_iter:
            retval += sub.evaluate(*args, **kwargs)
        return retval

    def __setitem__(self, key, value):
        self.data[key] = value
        self.recompute()

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]
        self.recompute()

    def __iter__(self):
        yield from self.data

    def items(self):
        yield from self.data.items()

    def keys(self):
        yield from self.data.keys()

    def values(self):
        yield from self.data.values()


class Basis(ParameterDependent):
    """A special superclass for basis objects.  This should be used
    together with the 'Bases' dictionary, which will keep the index
    attributes correct.
    """

    name: str
    start: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.shape[0]

    @property
    def indices(self):
        return np.arange(self.start, self.start + len(self))
