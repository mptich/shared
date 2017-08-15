# Utilities to handle GIS files

# Copyright (C) 2016-2017  Author: Misha Orel
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


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

