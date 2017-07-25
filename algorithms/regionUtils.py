# Utilities to process regions in 2D matrix. Each region is filled with corresponding non-ngative integer

import numpy as np
from shared.pyutils.tensorutils import *

def UtilGenerate4WayShifted(matPadded):
    return np.stack([matPadded[:-2,1:-1], matPadded[2:,1:-1], matPadded[1:-1,:-2], matPadded[1:-1,2:]], axis=-1)

def UtilRegionBoundaries(mat, levelCount=1):
    """
    Returns matrix of "levelCount" levels of region boundaries, where pixels on the boundary have ordinal
    numbers corresponding to their level, starting with 1, and non-border pixels are all 0
    :param mat: Input matrix, with the regions marked by non-negative integers
    :param levelCount: number of border levels to return
    :return: matrix with pixels on the boundary have ordinal numbers corresponding to their level, and non-border
    pixels are all 0
    """
    assert np.issubdtype(mat.dtype, np.integer) or np.issubdtype(mat.dtype, np.bool)
    matPadded = np.pad(mat, ((1,1), (1,1)), mode='symmetric')
    matShifted = UtilGenerate4WayShifted(matPadded)
    quadMat = np.repeat(mat[:,:,np.newaxis], 4, axis=2)
    matBound = np.where(np.any(np.logical_not(np.equal(matShifted, quadMat)), axis=2), 1, 0)
    if levelCount == 1:
        return matBound

    # Let's calculate secondary level etc
    matBoundPadded = np.pad(matBound, ((1,1), (1,1)), mode='constant', constant_values=0)
    matBoundView = matBoundPadded[1:-1, 1:-1]
    for level in range(2, levelCount+1):
        matBoundShifted = UtilGenerate4WayShifted(matBoundPadded)
        matBoundPadded[1:-1, 1:-1] = np.where(np.logical_and(matBoundView == 0, np.any(matBoundShifted == (level-1), axis=2)), \
                                level, matBoundView)

    return matBoundPadded[1:-1, 1:-1]


def UtilVisualizeRegions(mat):
    """
    returns image with regions in different colors
    Similar to cv2.applyColorMap, but returns colors that are far from each other :)
    :param mat: Matrix with regions as non-negative numbers
    :return: RGB image
    """

    assert np.issubdtype(mat.dtype, np.integer)

    idList = sorted(list(set(mat.flatten())))
    steps = 2
    while steps * steps * steps < len(idList):
        steps += 1

    step = 255 // (steps-1)
    colorPoints = range(0,255+1,step)
    colors = UtilCartesianMatrix3d(colorPoints, colorPoints, colorPoints).reshape(-1,3)[:len(idList),:]
    colorsXlate = np.zeros((idList[-1]+1,3), dtype=np.uint8)
    for idx, v in enumerate(idList):
        colorsXlate[v] = colors[idx]

    return colorsXlate[mat].astype(np.uint8)








