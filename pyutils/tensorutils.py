# Utilities for multidimensional arrays

__author__ = "Misha Orel"

import shared.pyutils.forwardCompat as forwardCompat
from shared.pyutils.utils import *


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


def UtilReflectCoordTensor(map, excludedArea=None):
    """
    Takes a tensor with values representing mapping coordinates, and replaces out-of-range values
    with reflections from edges
    :param img:
    :parameter excludeArea: if not None, then it denotes a rectangle that should not be mapped in thе reflection area
    :return:
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
        return ret

    minCoord = np.array(excludedArea[:n], dtype=np.float32)
    maxCoord = np.array(excludedArea[n:], dtype=np.float32)
    withinExclArea = np.logical_and(np.all(ret > minCoord, axis=n), np.all(ret <= maxCoord, axis=n))
    exclusion = np.logical_and(reflArea, withinExclArea)
    exclusion = np.repeat(exclusion, n).reshape(shape)
    return np.where(exclusion, minCoord, ret)







