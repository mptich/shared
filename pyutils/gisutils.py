# Utilities to handle GIS files
#
# Copyright (C) 2016-2018  Author: Misha Orel
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


import shared.pyutils.forwardCompat
from shared.pyutils.utils import UtilObject
import numpy as np
import gdal
import gdalconst


class BilFileBand(UtilObject):
    def __init__(self, img, parent, index):
        band = img.GetRasterBand(index + 1)
        self.parent = parent
        self.noDataValue = band.GetNoDataValue()
        self.data = band.ReadAsArray(0, 0, parent.ncol, parent.nrow)

    def value(self, x, y):
        parent = self.parent
        xn = int((x - parent.originX) / parent.pixelWidth)
        yn = int((y - parent.originY) / parent.pixelHeight)
        if (xn < 0 or yn < 0 or xn >= parent.ncol or yn >= parent.nrow):
            return None
        val = self.data[yn][xn]
        if val == self.noDataValue:
            val = None
        return val

class BilFile(UtilObject):
    """
    Reading BIL file
    This class has originally been taken from
    http://gis.stackexchange.com/questions/97828/reading-zipped-esri-bil-files-with-python
    and heavily modified
    """

    def __init__(self, bilFile, zipFile = None):
        if zipFile is not None:
            bilFile = "/vsizip/" + zipFile + "/" + bilFile
        gdal.GetDriverByName('EHdr').Register()
        img = gdal.Open(bilFile, gdalconst.GA_ReadOnly)
        self.bandCount = img.RasterCount
        self.ncol = img.RasterXSize
        self.nrow = img.RasterYSize
        geotransform = img.GetGeoTransform()
        self.originX = geotransform[0]
        self.originY = geotransform[3]
        self.pixelWidth = geotransform[1]
        self.pixelHeight = geotransform[5]
        self.bands = [BilFileBand(img, self, i) for i in range(self.bandCount)]

    def getBand(self, index):
        return self.bands[index]

    def getBandCount(self):
        return self.bandCount

