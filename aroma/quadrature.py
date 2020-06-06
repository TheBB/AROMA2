from itertools import product

import filebacked
import numpy as np
from numpy.polynomial.legendre import leggauss

from aroma.case import Parameters


class Quadrature(filebacked.FileBacked):

    parameters: Parameters
    weights: np.ndarray
    points: np.ndarray

    def __init__(self, parameters, weights, points):
        super().__init__()
        self.parameters = parameters
        self.weights = weights
        self.points = points

    def __len__(self):
        return len(self.weights)

    @classmethod
    def full(cls, parameters, npts=2, **kwargs):
        points, weights = [], []
        for name, (start, end) in parameters.ranges():
            pts, wts = leggauss(kwargs.get(name, npts))
            points.append((pts + 1)/2 * (end - start) + start)
            weights.append(wts/2 * (end - start))

        points = np.array(list(product(*points)))
        weights = np.array(list(map(np.product, product(*weights))))
        return cls(parameters, weights, points)

    @classmethod
    def empty(cls, parameters):
        nparams = sum(1 for _ in parameters.ranges())
        points = np.empty((0, nparams), dtype=float)
        weights = np.empty((0,), dtype=float)
        return cls(parameters, weights, points)

    def __iter__(self):
        for point in self.points:
            yield self.parameters.parameter(*point)

    def append(self, mu, weight=1.0):
        newpoint = [mu[name] for name, _ in self.parameters.ranges()]
        self.points = np.concatenate([self.points, np.array(newpoint).reshape(1,-1)])
        self.weights = np.concatenate([self.weights, [weight]])
