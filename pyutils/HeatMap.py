# Heat map manipulations
#
# Copyright (C) 2015-2018  Author: Misha Orel
#
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


import shared.pyutils.forwardCompat as forwardCompat
from shared.pyutils.utils import *
from shared.pyutils.imageutils import *
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

    def sum(self, heatMapList):
        lData = [h.data for h in heatMapList if h.valid]
        if lData:
            return HeatMap(sum(lData))
        return HeatMap(np.zeros((self.height, self.width), dtype=np.float32))

    def displayMultiple(self, fileName, colorDict):
        """
        Saves several superimposed heatmaps as an image
        :param colorDict: dictionary in form {'r': [list of red heatmaps], 'g': ..., 'b': ...}
        :param fileName: saves file name
        :return:
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for ind, key in enumerate(('r', 'g', 'b')):
            for hmap in colorDict.get(key, []):
                maxVal = np.max(hmap)
                maxVal = max(maxVal, UtilNumpyClippingValue(np.float32))
                img[:,:,ind] += hmap / maxVal

        maxVals = np.max(img, axis=(0,1)).clip(min=UtilNumpyClippingValue(np.float32))
        img = img * np.reciprocal(maxVals) * 255.0
        UtilArrayToImageFile(img, fileName)


class HeatMap(UtilObject):
    clippingValue = UtilNumpyClippingValue(np.float32)

    def __init__(self, arr, allowNegative=False):
        # Normalize
        assert (np.min(arr) >= 0.) or allowNegative
        self.weight = np.sum(arr)
        area = arr.shape[0] * arr.shape[1]
        if self.weight > self.clippingValue * area:
            self.data = arr / self.weight
            self.valid = True
        else:
            self.data = arr
            self.valid = False

    def pointDistance(self, hMapHlpr, yCoord, xCoord):
        return hMapHlpr.pointDistance(self, yCoord, xCoord)

    def maxCoord(self):
        assert self.valid
        return np.unravel_index(np.argmax(self.data), self.data.shape)

    def centroid(self):
        assert self.valid
        return UtilImageCentroid(self.data)

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

    def display(self, fileName):
        assert self.valid
        maxVal = np.max(self.data)
        dispMap = self.data / maxVal
        def _colorMap(lower, upper, dispMap):
            return ((dispMap - lower) / (upper - lower) * 255.).clip(min=0., max=255.)
        blue = _colorMap(0., 0.11, dispMap)
        red = _colorMap(0.11, 0.41, dispMap)
        green = _colorMap(0.41, 1.0, dispMap)
        img = np.stack([red, green, blue], axis=2)
        UtilArrayToImageFile(img, fileName)

    def getData(self):
        # Valid or not
        return np.copy(self.data)
