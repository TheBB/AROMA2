from functools import partial, lru_cache
from itertools import chain
from typing import Optional, Dict

from filebacked import FileBacked, FileBackedDict
from flexarrays import FlexArray
import numpy as np

from aroma.affine import ParameterDependent, ParameterConstant, Basis
from aroma.mufunc import MuFunc


class Parameter(FileBacked):

    name: str
    minimum: float
    maximum: float
    default: Optional[float]
    fixed: Optional[float]

    def __init__(self, name, minimum, maximum, default=None, fixed=None):
        super().__init__()
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.default = default
        self.fixed = fixed

    @property
    def mufunc(self):
        return MuFunc(self.name)


class Parameters(FileBackedDict[str, Parameter]):

    def add(self, name, *args, **kwargs):
        parameter = Parameter(name, *args, **kwargs)
        self[name] = parameter
        return parameter.mufunc

    def parameter(self, *args, **kwargs):
        retval, index = dict(), 0
        for param in self.values():
            if param.fixed is not None:
                retval[param.name] = param.fixed
                continue
            elif param.name in kwargs:
                retval[param.name] = kwargs[param.name]
            elif index < len(args):
                retval[param.name] = args[index]
            else:
                retval[param.name] = param.default
            index += 1
        return retval

    def ranges(self):
        for name, param in self.items():
            if param.fixed is None:
                yield (name, (param.minimum, param.maximum))


class Bases(FileBackedDict[str, Basis]):

    def recompute(self):
        start = 0
        for basis in self.values():
            basis.start = start
            start += len(basis)

    def __setitem__(self, key, value):
        value.name = key
        super().__setitem__(key, value)
        self.recompute()


class Case(FileBacked):

    name: str
    parameters: Parameters
    bases: Bases
    functions: Dict[str, ParameterDependent]
    constraints: FlexArray

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.parameters = Parameters()
        self.bases = Bases()
        self.functions = dict()

    def parameter(self, *args, **kwargs):
        return self.parameters.parameter(*args, **kwargs)

    def basis(self, name, mu, **kwargs):
        return self.bases[name](self, mu, **kwargs)

    @property
    def ndofs(self):
        return sum(len(basis) for basis in self.bases.values())

    @property
    def cons(self):
        try:
            return self.constraints
        except AttributeError:
            cons = FlexArray(ndim=1)
            for name, basis in self.bases.items():
                cons += FlexArray.vector(name, np.full((len(basis),), np.nan))
            self.constraints = cons
            return cons

    @cons.setter
    def cons(self, value):
        self.constraints = value

    def constrain(self, basisname, vector):
        cons = self.cons
        cons[basisname] = np.where(np.isnan(cons[basisname]), vector, cons[basisname])

    def __contains__(self, name):
        return name in self.functions

    @lru_cache
    def __getitem__(self, name):
        return partial(self.functions[name], self)

    def __setitem__(self, name, value):
        self.functions[name] = value


class HifiCase(Case):

    contractables: Dict[str, ParameterDependent]

    def __init__(self, name):
        super().__init__(name)
        self.contractables = dict()
        self.contractables['lift'] = ParameterConstant(FlexArray(ndim=1))

    def contractable(self, path, mu):
        func = self.contractables
        for component in path.split('/'):
            func = func[component]
        return func(self, mu)
