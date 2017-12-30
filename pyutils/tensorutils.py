# Utilities for multidimensional arrays
#
# Copyright (C) 2008-2017  Author: Misha Orel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

__author__ = "Misha Orel"

import shared.pyutils.forwardCompat as forwardCompat
from shared.pyutils.utils import *
import functools
import cv2


def UtilRandomSinFunc(shape, order, expectedStd, independentAxes=False):
    """
    Returns function made of multidimensional sin functions, with randomly selected phase and amplitude,
    max frequency by each axis is "order", mean = 0, std = std
    :param shape: dimensions of the output array
    :param order: max sin frequency along each axis. The higher the order, teh less correlated will be values
    in teh neighbor points. Order is either a number, or a tuple (one separate value per dimension)
    :param std: std of the output
    :return: f(Zn)
    """
    n = len(shape)
    angleMult = np.reciprocal(np.array(shape, dtype=np.float32)) * 2. * np.pi
    ret = np.zeros(shape=shape, dtype=np.float32)
    if not isinstance(order, tuple):
        order = tuple([order] * n)
    coordTensor = UtilCartesianMatrixDefault(*[x for x in shape]) * angleMult
    if not independentAxes:
        orderSet = UtilCartesianMatrixDefault(*[x+1 for x in order]).reshape(-1, n)[1:,:] # Exclude all 0s
    else:
        orderSet = [((0,) * y + (x,) + (0,) * (n-y-1)) for y in range(n) for x in range(1,order[y]+1)]
    for multipliers in orderSet:
        phase = np.random.uniform(low=-np.pi, high=np.pi)
        apmpl = np.random.uniform(low=0., high=1.)
        ret += np.sin(np.sum(coordTensor * multipliers, axis=n) + phase)
    std = np.std(ret)
    std = max(std, UtilNumpyClippingValue(np.float32))
    return expectedStd / std * ret


def UtilCartesianMatrix2d(arr1, arr2):
    """
    Converts [0,1], [2,4] into
    [0,2],[0,4]
    [1,2],[1,4]
    """
    arr1, arr2 = (np.array(x) if not isinstance(x, np.ndarray) else x for x in (arr1, arr2))
    assert len(arr1.shape) == len(arr2.shape) == 1
    transp = np.transpose([np.repeat(arr1, len(arr2)), np.tile(arr2, len(arr1))])
    return transp.reshape((len(arr1), len(arr2), 2))


def UtilCartesianMatrix3d(arr1, arr2, arr3):
    """
    Converts [0,1], [2,4], [8,9] into
    [[0,2,8],[0,2,9]], [[0,4,8], [0,4,9]]
    [[1,2,8],[1,2,9]], [[1,4,8],[1,4,9]]
    """
    arr1, arr2, arr3 = (np.array(x) if not isinstance(x, np.ndarray) else x for x in (arr1, arr2, arr3))
    assert len(arr1.shape) == len(arr2.shape) == len(arr3.shape) == 1
    transp = np.transpose([np.repeat(arr1, len(arr2)*len(arr3)), \
                           np.repeat(np.tile(arr2, len(arr1)), len(arr3)), \
                           np.tile(arr3, len(arr1)*len(arr2))])
    return transp.reshape((len(arr1), len(arr2), len(arr3), 3))


def UtilCartesianMatrix(arr1, arr2=None, arr3=None):
    if arr2 is None:
        return np.array(arr1).reshape(-1,1)
    elif arr3 is None:
        return UtilCartesianMatrix2d(arr1, arr2)
    return UtilCartesianMatrix3d(arr1, arr2, arr3)


@UtilStaticVars(cached={})
def UtilCartesianMatrixDefault(size1, size2=None, size3=None):
    """
    Returns potentially cached Cartesian matrix of range(size1), range(size2), range(size3)
    """
    tup = (size1, size2, size3)
    if tup in UtilCartesianMatrixDefault.cached:
        return UtilCartesianMatrixDefault.cached[tup]
    if size2 is None:
        ret = np.array(range(size1)).reshape(-1,1)
    elif size3 is None:
        ret = UtilCartesianMatrix2d(range(size1), range(size2))
    else:
        ret = UtilCartesianMatrix3d(range(size1), range(size2), range(size3))
    UtilCartesianMatrixDefault.cached[tup] = ret
    return ret


def UtilReflectCoordTensorWithExclusion(map, excludedArea=None):
    """
    Takes a tensor with values representing mapping coordinates, and replaces out-of-range values
    with reflections from edges
    :param img:
    :parameter excludeArea: if not None, then it denotes a rectangle that should not be mapped in thе reflection area
    :return: tuple(reflected map, boolean object exclusion pixels(shape=map.shape[:-1]))
    """
    def _reflectMap(singleVarMap, size):
        tiledMap = (singleVarMap / size).astype(np.int)
        singleVarMap = singleVarMap - tiledMap * size
        reflectMap = np.bitwise_and(tiledMap, 1)
        return np.where(reflectMap, size - singleVarMap, singleVarMap)
    n = len(map.shape) - 1
    assert map.shape[n] == n
    shape = map.shape
    reshapedMap = map.reshape((-1, n))

    if excludedArea is not None:
        reflArea = np.any( \
            np.stack([np.logical_or(reshapedMap[:,i] < 0, reshapedMap[:,i] >= shape[i]) \
            for i in range(n)], axis=1), axis=1).reshape(shape[:-1])

    reshapedMap = np.abs(reshapedMap)
    ret = np.stack([_reflectMap(reshapedMap[:, i], shape[i]) for i in range(n)], axis=1).reshape(shape)

    if excludedArea is None:
        return (ret, None)

    minCoord = np.array(excludedArea[:n], dtype=np.float32)
    maxCoord = np.array(excludedArea[n:], dtype=np.float32)
    withinExclArea = np.logical_and(np.all(ret > minCoord, axis=n), np.all(ret <= maxCoord, axis=n))
    exclusion = np.logical_and(reflArea, withinExclArea)
    return (ret, exclusion)


def UtilReflectCoordTensor(map):
    return UtilReflectCoordTensorWithExclusion(map)[0]


def UtilAvg2DPointsDistance(pts1, pts2):
    if (pts1 is None) or (pts2 is None):
        return (None, None, None, None)
    assert pts1.shape == pts2.shape
    assert pts1.shape[-1] == 2
    diff = pts1 - pts2
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    worstDist = np.max(dist)
    worstIndex = np.argmax(dist)
    meanDist = np.mean(dist)
    sigmaDist = np.sqrt(np.mean(dist * dist))
    return (meanDist, sigmaDist, worstDist, worstIndex)

def UtilNumpyRle(arr):
    """
    Numpy RLE algo
    :param arr: 1 dimensional array
    :return: tuple: start/end index of flat intervals, values at those intervals
    """
    assert len(arr.shape) == 1
    length = arr.shape[0]
    if length < 2:
        return (None, None)
    changes = np.where(arr[1:] != arr[:-1])[0]
    changes = np.append(0, changes + 1)
    values = arr[changes]
    intervals = np.stack([changes, np.append(changes[1:], length)], axis=1).astype(np.int)
    return (intervals, values)


def UtilNumpyExpandRle(intervals, values):
    """
    Reverse to UtilNumpyRle
    :param intervals: ndarray of shape (N, 2), with intervals
    :param values: ndarray of shape (N,), with values for teh intervals
    :return: 1 dimensional array
    """
    assert intervals.shape[0] == values.shape[0]
    ret = np.empty((intervals[-1][1] - intervals[0][0],), dtype=values.dtype)
    for ind, pair in enumerate(intervals):
        ret[pair[0]:pair[1]] = values[ind]
    return ret


def UtilImageCentroid(image):
    m = cv2.moments(image)
    area = m['m00'] + UtilNumpyClippingValue(np.float32)
    return (m['m01']/area, m['m10']/area)


def UtilImageCovarMatrix(image):
    """
    :param image:
    :return: returns centralized normalized moments
    """
    m = cv2.moments(image)
    area = m['m00'] + UtilNumpyClippingValue(np.float32)
    return (m['mu02']/area, m['mu11']/area, m['mu20']/area)


def UtilCartesianToPolar(arr):
    assert arr.shape[-1] == 2
    flatArr = arr.reshape(-1, 2)
    return np.stack([np.linalg.norm(arr, axis=-1), np.arctan2(flatArr[:,0],flatArr[:,1]).reshape(arr.shape[:-1])],
                    axis=-1)


def UtilPolarToCartesian(arr):
    assert arr.shape[-1] == 2
    flatArr = arr.reshape(-1, 2)
    amp = flatArr[:,0]
    angle = flatArr[:,1]
    return np.stack([amp * np.sin(angle), amp * np.cos(angle)], axis=1).reshape(arr.shape)


# It is expensive first time, so implement it as a function
def UtilNumpyClippingValue(dtype):
    info = np.finfo(dtype=dtype)
    return info.tiny * 10.


def UtilNumpyEntryItemSize(typeShapeTuple):
    """
    Calculates size of a numpy object
    :param typeShapeTuple: (numpy type, numpy shape)
    :return: size
    """
    return np.dtype(typeShapeTuple[0]).itemsize * functools.reduce(lambda x, y: x*y, typeShapeTuple[1])


def UtilNumpyEntriesSize(typeShapeList):
    """
    Calculates size of several numpy objects
    :param typeShapeList: list of tuples (numpy type, numpy shape)
    :return: size
    """
    return sum([UtilNumpyEntryItemSize(x) for x in typeShapeList])


def UtilRandomOptimChoice(array, selectCount):
    """
    Selects randomly from the array, providing the most uniform selection possible
    :param array: 1D array
    :param selectCount: how many entries to select
    :return: array of siz eselectCount
    """
    assert len(array.shape) == 1
    repeatCount = selectCount // array.shape[0]
    resCount = selectCount % array.shape[0]
    ret = np.repeat(array, repeatCount)
    ret = np.append(ret, np.random.choice(array, size = resCount, replace = False))
    return np.random.permutation(ret)


class QuickNumpyAppend(UtilObject):
    """
    Class to speed up (by caching) numpy.append operation
    """

    def __init__(self, arr=None, axis=None):
        self.axis = axis
        if arr is None:
            self.cache = []
        else:
            self.cache = [arr]

    def append(self, newArr):
        self.cache.append(newArr)

    def finalize(self):
        if self.axis is None:
            self.cache = [x.flatten() for x in self.cache]
            axis = 0
        else:
            axis = self.axis
        return np.concatenate(self.cache, axis=axis)





