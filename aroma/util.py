from collections import namedtuple
from functools import partial, singledispatch
from itertools import chain, product
from operator import attrgetter
from typing import List, Dict

import filebacked
import numpy as np
from numpy import newaxis as _
import scipy.sparse


def broadcast_shapes(args):
    """Compute the shape resulting from broadcasting over the shapes of
    all arguments, according to standard Numpy rules.
    """

    shapes = [arg.shape for arg in args]
    max_ndim = max(len(shape) for shape in shapes)
    shapes = np.array([(1,) * (max_ndim - len(shape)) + shape for shape in shapes])

    result = []
    for col in shapes.T:
        lengths = set(c for c in col if c != 1)
        assert len(lengths) <= 1
        if not lengths:
            result.append(1)
        else:
            result.append(next(iter(lengths)))
    return tuple(result)


def tuple_union(tuples):
    """Compute union of tuples as if they were sets."""
    retval = set()
    retval.update(chain.from_iterable(tuples))
    return tuple(retval)


def dependency_union(*args):
    """Compute the union of all dependencies."""
    return tuple_union(arg.dependencies for arg in args)


class StringlyFileBacked(filebacked.FileBackedBase):
    """Superclass for persisting an object in string form.

    Subclasses must implement the __repr__ method for encoding the
    object as a string, and must support evaluating the resulting
    string in a namespace consisting only of that class.
    """

    def allow_lazy(self):
        return False

    def write(self, group, **kwargs):
        super().write(group)
        filebacked.write(group, self.__class__.__name__, repr(self), str, **kwargs)

    def _read(self, **kwargs):
        classname = self.__class__.__name__
        code = filebacked.read(self.__filebacked_group__[classname], str, **kwargs)
        obj = eval(code, {}, {classname: self.__class__})
        self.__dict__.update(obj.__dict__)
