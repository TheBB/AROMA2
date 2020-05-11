from functools import partial
from collections import namedtuple
from itertools import chain

import filebacked
import numpy as np
from numpy import newaxis as _
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


def apply_contraction(obj, names, contract):
    """Apply a contraction over a multidimensional array-like object.

    The contraction must be a tuple of either None (no contraction) or
    a vector with the correct length, for each dimension of the
    object.
    """

    axes = []
    newnames = []
    for i, (name, cont) in enumerate(zip(names, contract)):
        if cont is None:
            newnames.append(name)
            continue
        assert cont.ndim == 1
        for __ in range(i):
            cont = cont[_,...]
        while cont.ndim < obj.ndim:
            cont = cont[...,_]
        obj = obj * cont
        axes.append(i)
    return obj.sum(tuple(axes)), tuple(newnames)


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


class NamedBlock:

    def __init__(self, names, obj):
        self.names = names
        self.obj = obj

    @property
    def T(self):
        return NamedBlock(self.names[::-1], self.obj.T)


class NamedBlocks:

    def __init__(self, names, fill=0.0):
        self.fill = fill
        self.name_to_index = [
            {name: i for i, (name, _) in enumerate(namespec)}
            for namespec in names
        ]
        self.index_to_size = [
            [length for (_, length) in namespec]
            for namespec in names
        ]

        s = tuple(len(sizes) for sizes in self.index_to_size)
        self.blocks = np.zeros(tuple(len(sizes) for sizes in self.index_to_size), dtype=object)

    def __getitem__(self, key):
        index = tuple(self.name_to_index[i][k] for i, k in enumerate(key))
        return self.blocks[index]

    def __setitem__(self, key, value):
        index = tuple(self.name_to_index[i][k] for i, k in enumerate(key))
        self.blocks[index] = value

    def __iadd__(self, block):
        assert isinstance(block, NamedBlock)
        self[block.names] += block.obj
        return self

    def substitute_zeros(self, func):
        for index, v in np.ndenumerate(self.blocks):
            if isinstance(v, int) and v == 0:
                shape = tuple(self.index_to_size[i][j] for i, j in enumerate(index))
                self.blocks[index] = func(shape)

    def realize(self):
        # Sparse path
        if any(isinstance(elt, sparse.spmatrix) for elt in self.blocks.flat):
            assert self.fill == 0
            self.substitute_zeros(sparse.coo_matrix)
            return sparse.bmat(self.blocks, format='csr')

        # Dense path
        self.substitute_zeros(partial(np.full, fill_value=self.fill))
        return np.block(self.blocks.tolist())


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
