import numpy as np

from aroma.util import StringlyFileBacked


_mufuncs = {
    '__sin': lambda x: np.sin(x),
    '__cos': lambda x: np.cos(x),
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / y,
    '**': lambda x, y: x ** y,
}


def _wrap(func):
    """Helper decorator for wrapping arguments as MuFunc objects."""

    def ret(*args):
        allowed = (MuFunc, str)
        if not all(isinstance(arg, allowed) or np.isscalar(arg) for arg in args):
            return NotImplemented
        new_args = [arg if isinstance(arg, MuFunc) else MuFunc(arg) for arg in args]
        return func(*new_args)
    return ret


class MuFunc(StringlyFileBacked):
    """MuFunc represents a scalar function of a parameter vector.

    A MuFunc has an operator and any number of operands.  The operator
    may be a string:

    - standard arithmetic: +, -, *, / or **, each of which require
      precisely two operands
    - a limited set of supported functions: __sin and __cos, each of
      which require precisely one operand
    - any other string, which implies evaluation of a named parameter
      component

    The operator may also be a constant scalar.

    Operands must themselves be MuFunc instances. MuFuncs are
    filebacked-compatible, and are persisted as a string.
    """

    __array_priority__ = 1.0

    def __init__(self, *args):
        self.oper, *self.operands = args

    @property
    def dependencies(self):
        """Return the parameters on which this function depends, as a set of
        strings.
        """
        subdeps = set()
        for op in self.operands:
            subdeps |= op.deps
        if not isinstance(self.oper, str):
            subdeps.add(self.oper)
        return subdeps

    def __repr__(self):
        opers = ', '.join(repr(op) for op in self.operands)
        return f"MuFunc({repr(self.oper)}, {opers})"

    def __call__(self, p):
        try:
            _mufuncs[self.oper](*(op(p) for op in self.operands))
        except KeyError:
            pass
        if isinstance(self.oper, str):
            return p[self.oper]
        return self.oper

    @_wrap
    def __add__(self, other):
        return MuFunc('+', self, other)

    @_wrap
    def __radd__(self, other):
        return MuFunc('+', other, self)

    @_wrap
    def __sub__(self, other):
        return MuFunc('-', self, other)

    @_wrap
    def __rsub__(self, other):
        return MuFunc('-', other, self)

    @_wrap
    def __mul__(self, other):
        return MuFunc('*', self, other)

    @_wrap
    def __rmul__(self, other):
        return MuFunc('*', other, self)

    @_wrap
    def __neg__(self):
        return MuFunc('-', mu(0.0), self)

    @_wrap
    def __pos__(self):
        return self

    @_wrap
    def __pow__(self, other):
        return MuFunc('**', self, other)

    @_wrap
    def __truediv__(self, other):
        return MuFunc('/', self, other)

    @_wrap
    def __rtruediv__(self, other):
        return MuFunc('/', other, self)

    def sin(self):
        return MuFunc('__sin', self)

    def cos(self):
        return MuFunc('__cos', self)
