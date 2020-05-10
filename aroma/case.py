from functools import partial, lru_cache
from itertools import chain
from typing import Optional, Dict

from filebacked import FileBacked, FileBackedDict

from aroma.affine import ParameterDependent, ParameterContainer
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


class Case(FileBacked):

    name: str
    parameters: Parameters
    geometry_func: ParameterDependent
    lhs: ParameterContainer
    rhs: ParameterContainer

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.parameters = Parameters()
        self.lhs = ParameterContainer()
        self.rhs = ParameterContainer()

    def parameter(self, *args, **kwargs):
        return self.parameters.parameter(*args, **kwargs)

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

    def function_search_path(self):
        return (self.lhs, self.rhs)


class HifiCase(Case):

    contractables: Dict[str, ParameterDependent]

    def __init__(self, name):
        super().__init__(name)
        self.contractables = dict()

    def function_search_path(self):
        return chain(super()._func_search(), (self.contractables,))


class NutilsCase(HifiCase):

    domain: object

    def __init__(self, name, domain):
        super().__init__(name)
        self.domain = domain
