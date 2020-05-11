from functools import partial, lru_cache
from itertools import chain
from typing import Optional, Dict

from filebacked import FileBacked, FileBackedDict
import numpy as np

from aroma.affine import ParameterDependent, Basis
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
    geometry_func: ParameterDependent
    functions: Dict[str, ParameterDependent]

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
    def geometry(self):
        return partial(self.geometry_func, self)

    @geometry.setter
    def geometry(self, value):
        self.geometry_func = value

    @lru_cache
    def __getitem__(self, name):
        for mapping in self.function_search_path():
            if name in mapping:
                return partial(mapping[name], self)
        raise KeyError(name)

    def __setitem__(self, name, value):
        self.functions[name] = value

    def function_search_path(self):
        return (self.functions,)


class HifiCase(Case):

    contractables: Dict[str, ParameterDependent]

    def __init__(self, name):
        super().__init__(name)
        self.contractables = dict()

    def function_search_path(self):
        return chain(super().function_search_path(), (self.contractables,))
