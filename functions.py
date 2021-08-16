import numpy as np
import numpy.random as rnd

from enum import Enum


class SampleType(Enum):
    LINE = 1
    SQUARE = 2
    TRIANGLE = 3
    SAW = 4
    DIAMOND = 5


def GMMsample(mu, var, p=None, n=1, seed=None):
    if seed is not None:
        rnd.seed(seed)

    m = np.array(mu)
    v = np.array(var)

    C = m.shape[0]  # number of components
    D = m.shape[1]  # number of dimensions

    # print C, D
    if p is None:
        p = np.ones(C)/C

    # print p

    X = np.zeros([n, D])
    for i in range(n):
        c = rnd.choice(C, p=p)
        X[i] = rnd.multivariate_normal(m[c], v)

    return X


def ShapeSample(sample_type, interval_min, interval_max, slope=1, bias=0, n=1, seed=None):
    if sample_type == SampleType.LINE:
        X = LineSample(interval_min, interval_max, slope, bias, n, seed)
    elif sample_type == SampleType.SQUARE:
        X = SquareSample(interval_min, interval_max, slope, bias, n, seed)
    elif sample_type == SampleType.TRIANGLE:
        X = TriangleSample(interval_min, interval_max, slope, bias, n, seed)
    elif sample_type == SampleType.SAW:
        X = SawSample(interval_min, interval_max, slope, bias, n, seed)
    elif sample_type == SampleType.DIAMOND:
        X = DiamondSample(interval_min, interval_max, slope, bias, n, seed)
    else:
        X = None
    return X


def LineSample(interval_min, interval_max, slope=1, bias=0, n=1, seed=None):
    if seed is not None:
        rnd.seed(seed)

    x = (interval_max - interval_min) * rnd.random(n) + interval_min
    y = slope * x + bias

    return np.stack([x, y], axis=1)


def SquareSample(interval_min, interval_max, slope=1, bias=0, n=1, seed=None):
    if seed is not None:
        rnd.seed(seed)

    x = (interval_max - interval_min) * rnd.random(n) + interval_min
    y = slope * np.power(x, 2) + bias

    return np.stack([x, y], axis=1)


def TriangleSample(interval_min, interval_max, slope=1, bias=0, n=1, seed=None):
    if seed is not None:
        rnd.seed(seed)

    interval_half = (interval_max - interval_min)/2.0 + interval_min
    x = (interval_max - interval_min) * rnd.random(n) + interval_min
    y = (x < interval_half) * (slope * x + bias) + (x >= interval_half) * \
        (slope * (interval_min + interval_max - x) + bias)

    return np.stack([x, y], axis=1)


def SawSample(interval_min, interval_max, slope=1, bias=0, n=1, seed=None):
    if seed is not None:
        rnd.seed(seed)

    interval_half = (interval_max - interval_min)/2.0 + interval_min
    x = (interval_max - interval_min) * rnd.random(n) + interval_min
    y = (x < interval_half) * (slope * x + bias) + (x >= interval_half) * \
        (slope * (x - interval_half + interval_min) + bias)

    return np.stack([x, y], axis=1)


def DiamondSample(interval_min, interval_max, slope=1, bias=0, n=1, seed=None):
    if seed is not None:
        rnd.seed(seed)

    interval_half = (interval_max - interval_min)/2.0 + interval_min
    x = (interval_max - interval_min) * rnd.random(n) + interval_min
    y_max = (x < interval_half) * (slope * x + bias) + (x >= interval_half) * \
        (slope * (interval_min + interval_max - x) + bias)
    y_min = (x < interval_half) * (slope * (2 * interval_min - x) + bias) + \
        (x >= interval_half) * (slope * (x + interval_min - interval_max) + bias)
    y = (y_max - y_min) * rnd.random(n) + y_min

    return np.stack([x, y], axis=1)
