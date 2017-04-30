# Heat map manipulations

import shared.pyutils.forwardCompat as forwardCompat
from shared.pyutils.utils import *
from scipy.ndimage import measurements as scipyMeasure
import math

class HeatMapHelper(UtilObject):
    """
    Class that speeds up HeatMap calculations. It is instantiated per (height, width) of the HeatMaps
    """
    def __init__(self, height, width):
        ind = np.indices((2*height-1, 2*width-1), dtype=np.float32)
        indY = ind[0,:,:]
        indX = ind[1,:,:]
        indY -= float(height-1)
        indX -= float(width-1)
        self.dist = np.sqrt(indX * indX + indY * indY)
        self.height = height
        self.width = width
        oneArray = np.empty((height, width), dtype=np.float32)
        oneArray.fill(1.0)
        self.flatHeatMap = HeatMap(oneArray)

    def pointDistance(self, hMap, yCoord, xCoord):
        """
        Calculates distance from a point to a HeatMap
        :param hMap: HeapMap
        :param yCoord: y coordinates of teh point
        :param xCoord: x coordinate of the point
        :return: distance, np.float32
        """
        height, width = (self.height, self.width)
        assert hMap.data.shape == (height, width)
        return np.sum(hMap.data * self.dist[height-1-yCoord:2*height-1-yCoord, width-1-xCoord:2*width-1-xCoord])

    def randomPointDistance(self, yCoord, xCoord):
        """
        Calculates a distance from a given point to a random HeatMap
        :param ycoord: y coordinates of the point
        :param xCoord: y coordinates of the point
        :return: distance, np.float32
        """
        return self.pointDistance(self.flatHeatMap, yCoord, xCoord)


class HeatMap(UtilObject):
    clippingValue = UtilNumpyClippingValue(np.float32)

    def __init__(self, arr):
        # Normalize
        self.weight = np.sum(arr)
        area = arr.shape[0] * arr.shape[1]
        if self.weight > self.clippingValue * area:
            self.data = arr / self.weight
            self.valid = True
        else:
            self.valid = False

    def pointDistance(self, hMapHlpr, yCoord, xCoord):
        return hMapHlpr.pointDistance(self, yCoord, xCoord)

    def maxCoord(self):
        assert self.valid
        return np.unravel_index(np.argmax(self.data), self.data.shape)

    def spread(self):
        assert self.valid
        sqData = self.data * self.data
        sqData = sqData / np.sum(sqData)
        yCenter, xCenter = scipyMeasure.center_of_mass(sqData)
        h,w = self.data.shape
        byY = np.repeat(range(h), w).reshape((h,w)).astype(np.float32) - yCenter
        byX = np.tile(range(w), h).reshape((h,w)).astype(np.float32) - xCenter
        byY *= byY
        byX *= byX
        return (np.sqrt(np.sum(sqData * byY)), np.sqrt(np.sum(sqData * byX)))

    def pointDistMax(self, yCoord, xCoord):
        y, x = self.maxCoord()
        dx = x - xCoord
        dy = y - yCoord
        return math.sqrt(dx*dx + dy*dy)