from contextlib import contextmanager
from typing import Dict, Optional, List

import filebacked
import numpy as np
from nutils import log

from aroma.util import FlexArray
from aroma.quadrature import Quadrature


class EnsembleBuilder(filebacked.FileBacked):

    quadrature: Quadrature
    snapshots: List[FlexArray]

    def __init__(self, quadrature):
        super().__init__()
        self.quadrature = quadrature
        self.snapshots = []

    def __len__(self):
        return len(self.quadrature)

    def __iter__(self):
        yield from self.quadrature

    def append(self, lhs, mu=None, weight=None):
        if mu is not None:
            assert len(self.snapshots) == len(self.quadrature)
            self.quadrature.append(mu, weight)
        assert len(self.snapshots) < len(self.quadrature)
        self.snapshots.append(lhs)

    def finalize(self):
        assert len(self.snapshots) == len(self.quadrature)
        return Ensemble(self.quadrature, np.array(self.snapshots))


class Ensemble(filebacked.FileBacked):

    quadrature: Quadrature
    snapshots: np.ndarray

    def __init__(self, quadrature, snapshots=None):
        super().__init__()
        self.quadrature = quadrature
        self.snapshots = snapshots

    @property
    def parameters(self):
        yield from self.quadrature

    @property
    def weights(self):
        return self.quadrature.weights

    def map(self, func, builder=EnsembleBuilder, name='snapshot', **kwargs):
        builder = builder(self.quadrature)
        for mu, arg in log.iter.plain(name, zip(builder, self.snapshots)):
            builder.append(func(mu, arg, **kwargs))
        return builder.finalize()
