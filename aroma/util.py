from itertools import chain

import filebacked
import numpy as np
import scipy.sparse as sparse


SCALARS = (
    float, np.float, np.float128, np.float64, np.float32, np.float16,
    int, np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
)


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


def apply_contraction(obj, contract):
    """Apply a contraction over a multidimensional array-like object.

    The contraction must be a tuple of either None (no contraction) or
    a vector with the correct length, for each dimension of the
    object.
    """

    axes = []
    for i, cont in enumerate(contract):
        if cont is None:
            continue
        assert cont.ndim == 1
        for __ in range(i):
            cont = cont[_,...]
        while cont.ndim < obj.ndim:
            cont = cont[...,_]
        obj = obj * cont
        axes.append(i)
    return obj.sum(tuple(axes))


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


class COOSparse(sparse.coo_matrix):
    """Subclass of Scipy's COO sparse matrix that supports addition."""

    def __add__(self, other):
        if not isinstance(other, COOSparse):
            return super().__add__(other)
        assert self.shape == other.shape
        newdata = np.concatenate((self.data, other.data))
        newrow = np.concatenate((self.row, other.row))
        newcol = np.concatenate((self.col, other.col))
        return COOSparse((newdata, (newrow, newcol)), shape=self.shape)

    def __iadd__(self, other):
        if not isinstance(other, COOSparse):
            return super().__add__(other)
        assert self.shape == other.shape
        self.data = np.concatenate((self.data, other.data))
        self.row = np.concatenate((self.row, other.row))
        self.col = np.concatenate((self.col, other.col))
        return self
