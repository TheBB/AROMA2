from __future__ import annotations
from typing import Dict, Tuple, Optional, Union

from filebacked import FileBacked
from flexarrays import FlexArray, zero_sentinel
import numpy as np
import scipy.sparse as sparselib

from aroma.util import broadcast_shapes, dependency_union
from aroma.mufunc import MuFunc


class ParametrizedFlexArray(FlexArray):

    def __call__(self, case, mu, block=None, **kwargs):
        retval = FlexArray()
        if block is not None:
            retval += self[block](case, mu, block=block, **kwargs)
        else:
            for index, func in self.items():
                retval += func(case, mu, block=index, **kwargs)
        return retval


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
            contract = tuple(
                case.contractables[c](case, mu) if isinstance(c, str) else c
                for c in contract
            )
            assert all(not isinstance(c, str) for c in contract)
            retval = self.evaluate(case, mu, contract, **kwargs)

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


class ParameterConstant(ParameterDependent):
    """Helper class for wrapping a constant as a ParameterDependent
    object.
    """

    obj: Union[np.ndarray, sparselib.spmatrix]

    def __init__(self, obj, **kwargs):
        super().__init__(obj.shape, **kwargs)
        self.obj = obj

    def evaluate(self, case, mu, contract, block, **kwargs):
        return FlexArray.single(block, self.obj).contract_many(contract)


def constant(array: FlexArray, **kwargs):
    retval = ParametrizedFlexArray()
    for index, value in array.items():
        retval.add(index, ParameterConstant(value, **kwargs))
    return retval


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
        raise NotImplementedError

    @property
    def indices(self):
        return np.arange(self.start, self.start + len(self))
