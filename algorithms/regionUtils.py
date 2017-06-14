# Utilities to process regions in 2D matrix. Each region is filled with corresponding non-ngative integer

import numpy as np

def _generate4WayShifted(matPadded):
    return np.dstack([matPadded[:-2,1:-1], matPadded[2:,1:-1], matPadded[1:-1,:-2], matPadded[1:-1,2:]])

def UtilRegionBoundaries(mat, levelCount=1):
    """
    Returns matrix of "levelCount" levels of region boundaries, where pixels on the boundary have ordinal
    numbers corresponding to their level, starting with 1, and non-border pixels are all 0
    :param mat: Input matrix, with the regions marked by non-negative integers
    :param levelCount: number of border levels to return
    :return: matrix with pixels on the boundary have ordinal numbers corresponding to their level, and non-border
    pixels are all 0
    """
    matPadded = np.pad(mat, ((1,1), (1,1)), mode='symmetric')
    matShifted = _generate4WayShifted(matPadded)
    quadMat = np.repeat(mat[:,:,np.newaxis], 4, axis=2)
    matBound = np.where(np.any(np.logical_not(np.equal(matShifted, quadMat)), axis=2), 1, 0)
    if levelCount == 1:
        return matBound

    # Let's calculate secondary level etc
    matBoundPadded = np.pad(matBound, ((1,1), (1,1)), mode='constant', constant_values=0)
    for level in range(2, levelCount+1):
        matBoundShifted = _generate4WayShifted(matBoundPadded)
        matBoundPadded[1:-1, 1:-1] = np.where(np.any(matBoundShifted == (level-1), axis=2), level, 0)

    return matBoundPadded[1:-1, 1:-1]






