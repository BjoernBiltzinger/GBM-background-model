from numba import guvectorize, njit, float64, int64, prange, jit
import numpy as np
from math import floor

"""
Partly taken from http://didattica.unibocconi.it/mypage/upload/49183_20180615_035144_INTERPOLATION.PY
"""


@jit(nopython=True, cache=True)
def _interpolation_search(x, z):
    """
    Interpolation search: locate z on grid x.
    Typically faster than binary search.
    """
    n = len(x)
    assert n > 1
    if z < x[1] or n == 2:
        return 0
    elif z >= x[-2]:
        return n - 2
    imin = 0
    imax = n - 1
    while (imax - imin) > 1:
        s = (z - x[imin]) / (x[imax] - x[imin])
        j = imin + floor((imax - imin) * s)
        if z >= x[j + 1]:
            imin = j + 1
        elif z < x[j]:
            imax = j
        else:
            return j
    return imin


@njit
def _locate(xn, x):
    index = np.zeros(len(xn))
    theta = np.zeros(len(xn))
    for i in range(len(xn)):
        j = int(_interpolation_search(x, xn[i]))
        index[i] = j
        theta[i] = (xn[i] - x[j]) / (x[j + 1] - x[j])
    return index, theta


@njit("float64[:,:,:](int64[:], float64[:], float64[:,:,:])", parallel=True, cache=True)
def _linear_numba(index, theta, y):
    """
    Computes linear interpolation for an array y. Uses numba.
    """
    yn = np.zeros((index.shape[0], y.shape[1], y.shape[2]))
    for j in prange(yn.shape[0]):
        yn[j] = (1 - theta[j]) * y[index[j], ...] + theta[j] * y[index[j] + 1, ...]
    return yn


def _linear_numpy(index, theta, y):
    """
    Computes linear interpolation for an array y.
    Vectorized Numpy.
    """
    yt1 = y[index, ...]
    yt2 = y[index + 1, ...]
    return ((1 - theta) * yt1.T + theta * yt2.T).T


class Interp1D(object):
    def __init__(self, xn, x, assume_sorted=True):
        """
        Interp1D: univariate fast linear interpolation
        :param x: interpolation nodes (N-D numpy array of size n).
        :param xn: points where to interpolate (N-D numpy array of size m)
        :param assume_sorted = False: if True, x is assumed to be sorted.
        """
        xn = np.asarray(xn).ravel()
        x = np.asarray(x).ravel()
        self._xn_size = xn.size
        self._x_size = x.size
        #  Sanity check
        if self._x_size < 2:
            raise ValueError("At least two nodes are needed for interpolation")
        if not isinstance(assume_sorted, bool):
            raise ValueError("assume_sorted must be a boolean")
        #  Sort x if needed
        if assume_sorted:
            self._argsort = None
        else:
            self._argsort = np.argsort(x)
            x = np.take(x, self._argsort)
        #  Locate xn on the grid x
        self._index, self._theta = _locate(xn, x)
        self._set_optimal_interpolation()

    def _set_optimal_interpolation(self):
        # if val>10:
        #    self._linear = _linear_numba
        # else:
        self._linear = _linear_numpy

    def __call__(self, y):
        """
        Get interpolation for defined nodes x and wanted xn for given y's at x's
        :param y: function values at nodes x
        :return: interpolated values
        """
        #  Sanity check
        if y.shape[0] != self._x_size:
            raise ValueError("Shape of y does not fit to shape of x")

        # Compute linear interpolation
        yn = self._linear(self._index, self._theta, y)
        return yn.reshape(-1, 2, y.shape[1], y.shape[2])
