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


@singledispatch
def contract_helper(obj, contract, axis):
    raise NotImplementedError

@contract_helper.register(np.ndarray)
def _(obj, contract, axis):
    newshape = [1] * obj.ndim
    newshape[axis] = len(contract)
    contract = contract.reshape(newshape)
    obj = (obj * contract).sum((axis,))
    return obj

@contract_helper.register(scipy.sparse.spmatrix)
def _(obj, contract, axis):
    if axis == 1:
        return obj @ contract
    elif axis == 0:
        return contract @ obj.T


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


class ZeroSentinel:

    @property
    def T(self):
        return self

    def __iadd__(self, other):
        return other

    def __add__(self, other):
        return other

    def __isub__(self, other):
        return -other

    def __sub__(self, other):
        return -other

    def __mul__(self, other):
        return self

    def __imul__(self, other):
        return self

    def __div__(self, other):
        return self

    def __idiv__(self, other):
        return self

zero_sentinel = ZeroSentinel()


class FlexArray:

    # An array of object dtype holding all the blocks, or the zero
    # integer if no data
    blocks: np.ndarray

    # A dictionary holding the length of each known block by name
    sizes: Dict[str, int]

    # A list holding the numerical block index of each named block on
    # each axis, e.g. if axis_indices[3]['a'] == 1 then data for block
    # 'a' on the fourth axis is stored in index 1
    axis_indices: List[Dict[str, int]]

    def __init__(self, *components, ndim=None):
        # If any components are given, determine number of dimensions
        # automatically
        if components:
            ndim = len(components[0][0])

        # Initialize data structures
        self.blocks = np.full((0,) * ndim, zero_sentinel, dtype=object)
        self.sizes = dict()
        self.axis_indices = [dict() for _ in range(ndim)]

        # Iteratively add-in each component
        for index, value in components:
            self.add_component(index, value)

    @classmethod
    def vector(cls, name, value):
        """Simplified constructor for a single-block vector."""
        return cls(((name,), value))

    @classmethod
    def single(cls, names, value):
        """Simplified constructor for a single-block array."""
        return cls((names, value))

    @classmethod
    def raw(cls, blocks, sizes, indices):
        """Internal-use constructor that creates a new FlexArray with direct
        access to the internal data structure.
        """
        newobj = cls.__new__(cls)
        newobj.blocks = blocks
        newobj.sizes = sizes
        newobj.axis_indices = indices
        return newobj

    @property
    def ndim(self):
        return self.blocks.ndim

    def copy(self, deep=True):
        """Create a copy of this FlexArray object. If 'deep' is true, the
        blocks will also be copied.
        """
        if deep:
            copier = lambda x: x.copy() if hasattr(x, 'copy') else x
        else:
            copier = lambda x: x
        newobj = FlexArray(ndim=self.ndim)
        for index, block in self.items():
            newobj.add_component(index, copier(block))
        return newobj

    def ranges(self, names):
        """Utility method for converting an ordered list of names to a list of
        indexing ranges.
        """
        previous, ranges = 0, []
        for name in names:
            ranges.append(np.arange(previous, previous + self.sizes[name]))
            previous += self.sizes[name]
        return ranges

    def compatible(self, blocknames, array):
        """Create a new FlexArray representation of the given array with the
        given block names. The new FlexArray is compatible with
        'self', in the sense that blocks with the same name must have
        the same length, and that after

            temp = flex.compatible(names, old)
            new = temp.realize(names)

        then 'new' and 'old' should be equal.
        """
        indexranges = [self.ranges(names) for names in blocknames]
        blockshape = tuple(len(names) for names in blocknames)
        newblocks = np.full(blockshape, zero_sentinel, dtype=object)
        for blockindex in np.ndindex(newblocks.shape):
            ranges = (indexranges[axis][j] for axis, j in enumerate(blockindex))
            newblocks[blockindex] = array[np.ix_(*ranges)]

        newindices = [{name: i for i, name in enumerate(names)} for names in blocknames]
        return type(self).raw(newblocks, self.sizes.copy(), newindices)

    @property
    def T(self):
        """Return the transposed array."""
        if self.ndim == 1:
            return self
        newblocks = np.vectorize(attrgetter('T'))(self.blocks.T)
        newsizes = self.sizes.copy()
        newindices = [indices.copy() for indices in self.axis_indices[::-1]]
        return type(self).raw(newblocks, newsizes, newindices)

    def only(self):
        """Return the index tuple and block contents, asserting that the array
        consists only of a single block.
        """
        assert self.blocks.size == 1
        index = tuple(next(iter(indices.keys())) for indices in self.axis_indices)
        return self.blocks.flat[0], index

    def __getitem__(self, names):
        """Get a block by index."""
        if isinstance(names, str):
            names = (names,)
        index = (indices[name] for name, indices in zip(names, self.axis_indices))
        retval = self.blocks[tuple(index)]
        assert retval is not zero_sentinel
        return retval

    def __setitem__(self, names, value):
        """Set a block by index."""
        if isinstance(names, str):
            names = (names,)
        index = (indices[name] for name, indices in zip(names, self.axis_indices))
        self.blocks[tuple(index)] = value

    def keys(self):
        """Iterate over nonzero blocks by index."""
        for name, _ in self.items():
            yield name

    def values(self):
        """Iterate over nonzero blocks by value."""
        for _, value in self.items():
            yield value

    def items(self):
        """Iterate over nonzero blocks by index and value."""
        names = [indices.keys() for indices in self.axis_indices]
        nums = [indices.values() for indices in self.axis_indices]
        for name, index in zip(product(*names), product(*nums)):
            value = self.blocks[index]
            if value is not zero_sentinel:
                yield name, value

    def add_component(self, index, value):
        """Add a new block to this array.  The index may be inexistent, in
        which case the internal data structures will be expanded, or
        refer to an existing block, in which case it will be added.
        """

        assert len(index) == self.ndim

        # Calculate the numerical block index axis-by-axis
        num_index = []
        for i, (name, length, indices) in enumerate(zip(index, value.shape, self.axis_indices)):

            # Check that the block has the correct size if we've seen
            # it before, or record the size if not
            assert self.sizes.setdefault(name, length) == length

            # If this is a new block for this axis, expand the block
            # arrray and record its index
            if name not in indices:
                indices[name] = len(indices)
                append_shape = self.blocks.shape[:i] + (1,) + self.blocks.shape[i+1:]
                append_array = np.full(append_shape, zero_sentinel, dtype=object)
                self.blocks = np.append(self.blocks, append_array, axis=i)

            num_index.append(indices[name])

        # We know the index, so just add it normally
        self.blocks[tuple(num_index)] += value

    def contract(self, contract, axis):
        """Contract the array by a single axis. The 'contract' argument should
        be either None or a one-dimensional FlexArray.
        """
        if contract is None:
            return self
        assert contract.ndim == 1
        newobj = FlexArray(ndim=self.ndim-1)
        for index, block in self.items():
            block = contract_helper(block, contract[index[axis]], axis)
            newindex = index[:axis] + index[axis+1:]
            newobj.add_component(newindex, block)
        return newobj

    def contract_many(self, contract):
        """Contract the array by multiple axes. The 'contract' argument must
        be a tuple of one-dimensional FlexArrays or None.
        """
        assert len(contract) == self.ndim
        retval = self.copy()
        for i, c in enumerate(reversed(contract)):
            axis = self.ndim - i - 1
            retval = retval.contract(c, axis)
        return retval

    def realize(self, *blocks, lengths=None, sparse=None):
        """Realize this block array as a true numpy array or scipy matrix,
        according to the ordering given by 'blocks'.

        If lengths is given, it should be a dictionary-like object
        mapping block name to size, verifying that the result is as
        expected.

        The 'sparse' parameter may be either True or a recognized
        Scipy sparsity format, False (returns a dense matrix) or None,
        in which case the sparsity of the returned object will be
        heuristically determined.
        """

        assert len(blocks) == self.ndim

        # If 'sparse' is not explicitly given, and we have a
        # two-dimensional block array with at least one sparse
        # component, return a sparse CSR matrix.  If 'sparse' is given
        # but does not specify the format, also use CSR.
        if sparse is None and self.ndim == 2:
            if any(isinstance(elt, scipy.sparse.spmatrix) for elt in self.blocks.flat):
                sparse = 'csr'
            else:
                sparse = False
        elif self.ndim != 2:
            sparse = False
        elif sparse is True:
            sparse = 'csr'
        elif not sparse:
            sparse = False

        # Optionally verify that lengths are what they should be
        if lengths is not None:
            assert all(self.sizes[name] == length for name, length in lengths.items())

        # Build the block list with the correct block numbering
        block_indices = []
        for blocklist, indices in zip(blocks, self.axis_indices):
            block_indices.append(tuple(indices[name] for name in blocklist))
        reordered_blocks = self.blocks[np.ix_(*block_indices)].copy()

        # Replace zero entries
        func = scipy.sparse.coo_matrix if sparse else partial(np.full, fill_value=0.0)
        for index, value in np.ndenumerate(reordered_blocks):
            if value is zero_sentinel:
                shape = tuple(self.sizes[blocks[i][j]] for i, j in enumerate(index))
                reordered_blocks[index] = func(shape)

        # Construct final return value
        if sparse:
            return scipy.sparse.bmat(reordered_blocks, format=sparse)
        return np.block(reordered_blocks.tolist())

    def __add__(self, other):
        newobj = self.copy()
        newobj += other
        return newobj

    def __iadd__(self, other):
        if np.isscalar(other):
            self.blocks += other
            return self
        if not isinstance(other, FlexArray):
            return NotImplemented
        assert self.ndim == other.ndim
        for name, value in other.items():
            self.add_component(name, value)
        return self

    def __sub__(self, other):
        newobj = self.copy()
        newobj -= other
        return newobj

    def __isub__(self, other):
        if np.isscalar(other):
            self.blocks -= other
            return self
        if not isinstance(other, FlexArray):
            return NotImplemented
        assert self.ndim == other.ndim
        for name, value in other.items():
            self.add_component(name, -value)
        return self

    def __mul__(self, other):
        newobj = self.copy()
        newobj *= other
        return newobj

    def __imul__(self, other):
        if np.isscalar(other):
            self.blocks *= other
            return self
        if not isinstance(other, FlexArray):
            return NotImplemented
        assert self.ndim == other.ndim
        common_names = set(self.keys()) & set(other.keys())
        for name in self.keys():
            if name not in common_names:
                self[name] = zero_sentinel
        for name, value in other.keys():
            if name in common_names:
                self[name] *= value
        self.sizes = {**self.sizes, **other.sizes}
        return self

    def __div__(self, other):
        newobj = self.copy()
        newobj /= other
        return newobj

    def __idiv__(self, other):
        assert np.isscalar(other)
        self.blocks *= other
        return
