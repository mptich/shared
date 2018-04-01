# Utilities for multidimensional arrays
#
# Copyright (C) 2008-2018  Author: Misha Orel

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


__author__ = "Misha Orel"

import shared.pyutils.forwardCompat as forwardCompat
from shared.pyutils.utils import *
import numpy as np
import functools
import cv2
from collections import Counter


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


def UtilCartesianMatrix(*arrayList):
    """
    Converts [0,1], [2,4], [8,9] into
    [[0,2,8],[0,2,9]], [[0,4,8], [0,4,9]]
    [[1,2,8],[1,2,9]], [[1,4,8],[1,4,9]]
    """
    shape = tuple([len(x) for x in arrayList])
    inds = UtilNumpyFlatIndices(shape)
    return np.stack([np.array(x)[inds[i]].reshape(shape) for i, x in enumerate(arrayList)], axis=len(arrayList))


@UtilStaticVars(cached={})
def UtilCartesianMatrixDefault(*sizeList):
    """
    Returns potentially cached Cartesian matrix of range(size1), range(size2), range(size3), ...
    """
    tup = tuple(sizeList)
    if tup in UtilCartesianMatrixDefault.cached:
        return UtilCartesianMatrixDefault.cached[tup]
    arrayList = [range(x) for x in tup]
    ret = UtilCartesianMatrix(*arrayList)
    UtilCartesianMatrixDefault.cached[tup] = ret
    return ret


def UtilNumpyFlatIndices(dims):
    """
    Presents indices of along each axes as a flat array. Output is a 2D array
    :param dims:
    :return:
    """
    inds = np.indices(dims)
    # Do not make it an ndarray, must be just a list to be correctly used as indices
    return [x.flatten() for x in inds[:]]


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


def UtilPerpendIntersectLine(linePt1, linePt2, perpPt):
    """
    Calculates coordinates of intersection between line (defined by points linePt1 and linePt2) and point perpPt
    :param linePt1:
    :param linePt2:
    :param perpPt:
    :return:
    """
    y1, x1 = linePt1
    y2, x2 = linePt2
    yp, xp = perpPt
    ydiff = y2 - y1
    xdiff = x2 - x1
    coef = ((y2 - y1) * (xp - x1) - (x2 - x1) * (yp - y1)) / (ydiff * ydiff + xdiff * xdiff)
    x = xp - coef * (y2 - y1)
    y = yp + coef * (x2 - x1)
    return np.array([y, x])


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


def UtilBalancedSetIndexes(labels):
    """
    Selects, based on 1D labels array, a set of indexes to generate a balanced, randomized set
    labels must be an array of unmutable (hashable) objects
    :param labels:
    :return: array of indexes into labels
    """
    d = Counter()
    for v in np.unique(labels):
        d[v] = np.sum(labels == v)
    mc = d.most_common()
    maxCount = mc[0][1]

    choices = []
    for v in d.keys():
        choices.append(UtilRandomOptimChoice(np.where(labels == v)[0], maxCount))
    choices = np.concatenate(choices, axis=0)
    return np.random.permutation(choices)


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

    def runLens(self):
        if self.axis is None:
            rl = [x.flatten().shape[0] for x in self.cache]
        else:
            rl = [x.shape[self.axis] for x in self.cache]
        return np.array(rl)

    def finalize(self):
        if self.axis is None:
            self.cache = [x.flatten() for x in self.cache]
            axis = 0
        else:
            axis = self.axis
        return np.concatenate(self.cache, axis=axis)


def UtilIntervalsToIndexes(intervals):
    """
    Converts 2D array of intervals into contiguous indexes
    :param intervals:
    :return:
    """
    ret = []
    for start, stop in intervals:
        ret += list(range(start, stop))
    return np.array(ret)


def UtilIntervalsToBooleans(intervals, upperLimit):
    """
    Converts 2D array of intervals into contiguous boolean array
    :param intervals:
    :param upperLimit: the size of output array
    :return:
    """
    ret = np.zeros((upperLimit,), dtype=np.bool)
    for start, stop in intervals:
        ret[start:stop] = True
    return ret


def UtilContigLengthesToAlternateBool(contigLens):
    """
    Transform contiguous lengths into numpy array of alternate booleans
    :param contigLens:
    :return:
    """
    totalLen = np.sum(contigLens)
    ret = np.empty((totalLen,), dtype=np.bool)
    start = 0
    val = True
    for len in contigLens:
        ret[start:start+len] = val
        val = not val
        start += len
    return ret


def UtilIntersectContigLens(contigLens1, contigLens2):
    """
    (2, 4, 2) x (1, 2, 2, 3) = (1, 1, 1, 2, 1, 2)
    :param contigLens1:
    :param contigLens2:
    :return:
    """
    boolAlt1 = UtilContigLengthesToAlternateBool(contigLens1)
    boolAlt2 = UtilContigLengthesToAlternateBool(contigLens2)
    assert boolAlt1.shape == boolAlt2.shape
    boolAlt = np.logical_xor(boolAlt1, boolAlt2)
    intervals, _ = UtilNumpyRle(boolAlt)
    return intervals[:, 1] - intervals[:, 0]


def UtilAdjustNumpyDims(arrayList):
    """
    Adjusts sizes of input arrays in the list to the minimum size by every axis
    :param arrayList:
    :return:
    """
    minShape = None
    for a in arrayList:
        if minShape is None:
            minShape = np.array(a.shape)
        else:
            arrShape = np.array(a.shape)
            minShape = np.where(arrShape < minShape, arrShape, minShape)

    minShape = tuple(minShape)
    inds = UtilNumpyFlatIndices(minShape)
    return [x[inds].reshape(minShape) for x in arrayList]


def UtilIsNumpyTypeFloat(numpyType):
    """
    The only reliable way I found for any float size
    :param numpyType:
    :return:
    """
    return str(numpyType)[:5] == 'float'


def UtilIsNumpyTypeInt(numpyType):
    """
    The only reliable way I found for any int size
    :param numpyType:
    :return:
    """
    return str(numpyType)[:3] == 'int'
