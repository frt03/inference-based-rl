import numpy as np
import scipy.stats
from typing import Sequence


def mean_or_nan(xs: Sequence[float]) -> float:
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def var_or_nan(xs: Sequence[float]) -> float:
    """Return its variance a non-empty sequence, numpy.nan for a empty one."""
    return np.var(xs) if xs else np.nan


def max_or_nan(xs: Sequence[float]) -> float:
    """Return its maximum a non-empty sequence, numpy.nan for a empty one."""
    return np.max(xs) if xs else np.nan


def min_or_nan(xs: Sequence[float]) -> float:
    """Return its maximum a non-empty sequence, numpy.nan for a empty one."""
    return np.min(xs) if xs else np.nan


def skew_or_nan(xs: Sequence[float]) -> float:
    """Return its skewness a non-empty sequence, numpy.nan for a empty one."""
    return scipy.stats.skew(xs) if xs else np.nan
