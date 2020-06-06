from contextlib import contextmanager
from typing import Dict, Optional, List

import filebacked
from flexarrays import FlexArray
import numpy as np
from nutils import log

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
        assert lhs.ndim == 1
        self.snapshots.append(lhs)

    def finalize(self):
        assert len(self.snapshots) == len(self.quadrature)
        snapshots = {
            key: np.array([snap[key] for snap in self.snapshots])
            for (key,) in self.snapshots[0].keys()
        }
        return Ensemble(self.quadrature, snapshots)


class Ensemble(filebacked.FileBacked):

    quadrature: Quadrature
    snapshots: Dict[str, np.ndarray]

    def __init__(self, quadrature, snapshots):
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
        for mu, arg in log.iter.plain(name, zip(builder, self)):
            builder.append(func(mu, arg, **kwargs))
        return builder.finalize()

    def __len__(self):
        return len(next(self.snapshots.iter()))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.snapshots[key]
        retval = FlexArray(ndim=1)
        for name, data in self.snapshots.items():
            retval.add(name, data[key])
        return retval
