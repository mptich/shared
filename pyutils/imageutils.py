# Image utilities
#
# Copyright (C) 2015-2017  Author: Misha Orel
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


import shared.pyutils.forwardCompat as forwardCompat
from shared.pyutils.utils import *
from shared.pyutils.tensorutils import *
from shared.algorithms.regionUtils import *
import PIL
_minPilVersion = '3.1'
if forwardCompat.VersionCompare(PIL.__version__, _minPilVersion) < 0:
    ValueError('OLD PIL VERSION %s, should be at least %s' % (PIL.__version__, _minPilVersion))
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageFilter
import scipy.ndimage.filters as scipyFilters
from scipy import interpolate
import scipy
_minScipyVersion = '0.14.0'
if forwardCompat.VersionCompare(scipy.__version__, _minScipyVersion) < 0:
    ValueError('OLD SCIPY VERSION %s, should be at least %s' % (scipy.__version__, _minScipyVersion))
import math
import sys
from sklearn import linear_model
import gc
import time
import operator
import collections
import csv
import cv2
_ExifHandledByCv2 = forwardCompat.VersionCompare(cv2.__version__, '3.1') >= 0


def UtilImageFileToArray(fileName, bgr=False, exifOrient=False):
    if _ExifHandledByCv2:
        exifOrient = False
    img = cv2.imread(fileName)
    if img is None:
        return None
    if (not bgr) and (len(img.shape) == 3) and (img.shape[2] == 3):
        img = np.flip(img, axis=2)
    if exifOrient:
        imgPil = Image.open(fileName)
        func = getattr(imgPil, '_getexif', lambda: None)
        exifDict = func()
        if exifDict is not None:
            img = UtilImageExifOrient(img, exifDict)
    return UtilImageToFloat(img)

def UtilArrayToImageFile(arr, fileName, jpgQuality=None, bgr=False):
    arr = UtilImageToInt(arr)
    if (not bgr) and (len(arr.shape) == 3) and (arr.shape[2] == 3):
        arr = np.flip(arr, axis=2)
    if (jpgQuality is None) or (os.path.splitext(fileName)[1].lower() not in ('.jpg', '.jpeg')):
        cv2.imwrite(fileName, arr)
    else:
        cv2.imwrite(fileName, arr, [cv2.CV_IMWRITE_JPEG_QUALITY, int(jpgQuality)])

def UtilFromRgbToGray(img):
    img = np.flip(img, axis=2)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

def UtilFromGrayToRgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return np.flip(img, axis=2).astype(np.float32)

def UtilImageToInt(img):
    return np.rint(img).astype(np.int).clip(min=0, max=255).astype(np.uint8)

def UtilImageToFloat(img):
    return img.astype(np.float32)

def UtilValidateImagePoint(shape, point):
    h,w=shape
    y,x=point
    return ((y >= 0) and (y < h) and (x >= 0) and (x < w))

def UtilValidatePointPosAbove(points1, points2, equal=False):
    y1 = np.max(np.array(points1)[:,0])
    y2 = np.min(np.array(points2)[:,0])
    if not equal:
        return (y2 > y1)
    else:
        return (y2 >= y1)

def UtilValidatePointPosRight(points1, points2, equal=False):
    x1 = np.min(np.array(points1)[:,1])
    x2 = np.max(np.array(points2)[:,1])
    if not equal:
        return (x1 > x2)
    else:
        return (x1 >= x2)

def UtilValidateBoundBox(shape, bb, margin=0):
    h,w=shape
    yMin,xMin,yMax,xMax = bb
    return ((yMin >= margin) and (yMin < h-margin) and (xMin >= margin) and (xMin < w-margin) and \
            (yMax <= h-margin) and (xMax <= w-margin) and (yMin < yMax) and (xMin < xMax))


def UtilVerticalScale(img, destHeight, destWidth, padValue=0, padMode='constant', interpMode='cubic', \
                      horPlacement=None):
    """
    Scales image in such a way that it is fit by height, and width either cropped or padded
    :param img: input image
    :param destHeight: desired height
    :param destWidth: desired width
    :param padValue: in case of padding, what value to use
    :param horPlacement: if not None, then it should be a floating in [0.,1.], determining how teh pictre is cropped
                         horizontally; horPlacement==None is equivalent to hotPlacement=0.5
    :return: tuple (Scaled image, scale ratio, left pad)
    """

    h,w = img.shape[:2]

    # Scale by height
    scale = destHeight / h
    realWidth = int(round(w * scale))
    img = UtilImageResize(img, destHeight, realWidth, interp=interpMode)

    # See if padding / clipping is needed
    if horPlacement is None:
        leftPadWidth = (destWidth - realWidth) // 2
    else:
        leftPadWidth = int(round((destWidth - realWidth) * horPlacement))
    rightPadWidth = destWidth - realWidth - leftPadWidth
    assert (leftPadWidth * rightPadWidth) >= 0
    if (leftPadWidth + rightPadWidth) > 0:
        padTuple = ((0, 0), (leftPadWidth, rightPadWidth))
        if len(img.shape) == 3:
            padTuple += ((0,0),)
        if padMode == 'constant':
            img = np.pad(img, padTuple, mode=padMode, constant_values=padValue)
        else:
            img = np.pad(img, padTuple, mode=padMode)
    else:
        img = img[:, -leftPadWidth:realWidth + rightPadWidth]

    assert img.shape[:2] == (destHeight, destWidth)
    return (img, scale, leftPadWidth)


def UtilImageResize(img, destHeight, destWidth, interp = 'cubic', asPicture = True):
    """
    It uses OpneCV
    :param img: input image
    :param destHeight: target height
    :param destWidth: target width
    :return:
    """
    h, w = img.shape[:2]
    assert interp in ('nearest', 'cubic', 'linear', 'lanczos')
    if interp == 'nearest':
        method = cv2.INTER_NEAREST
    elif (destHeight >= h) and (destWidth >= w):
        if interp == 'linear':
            method = cv2.INTER_LINEAR
        elif interp == 'cubic':
            method = cv2.INTER_CUBIC
        else:
            assert interp == 'lanczos'
            method = cv2.INTER_LANCZOS4
    else:
        method = cv2.INTER_AREA
    img = cv2.resize(img, (destWidth, destHeight), interpolation=method)
    if asPicture:
        img = img.clip(min=0., max=255.)
    return img

@UtilStaticVars(rotateDict={1:(0,0,0), 2:(0,0,1), 3:(0,1,1), 4:(0,1,0), 5:(1,0,0), 6:(1,0,1), 7:(1,1,1), 8:(1,1,0)})
def UtilImageExifOrient(img, exifTagsDict):
    """
    Rotates image according to what has been specified in the Exif structure
    :param img: Original image
    :param exifTagsDict: Exif structure
    :return: rotated image
    """
    # 0x112 is the orientation tag
    orientTuple = UtilImageExifOrient.rotateDict[exifTagsDict.get(0x112, 1)]
    if orientTuple[0]:
        axes = list(range(len(img.shape)))
        axes[0], axes[1] = (axes[1], axes[0])
        img = np.transpose(img, axes=axes)
    if orientTuple[1]:
        img = np.flip(img, axis=0)
    if orientTuple[2]:
        img = np.flip(img, axis=1)
    return img

def UtilStitchImagesHor(imgInputList, outImageName=None, padMode='constant', constValue=127., seamsList=None):
    imgList = []
    for img in imgInputList:
        if isinstance(img, str):
            img = UtilImageFileToArray(img)
        assert len(img.shape) == 3
        imgList.append(img)

    # Make them all of the same height
    maxHeight = max([x.shape[0] for x in imgList])
    eqImgList = []
    for img in imgList:
        img = np.pad(img, ((maxHeight - img.shape[0], 0), (0,0), (0,0)), mode=padMode, constant_values=constValue)
        eqImgList.append(img)

    width = 0
    for img in eqImgList:
        width += img.shape[1]
        if seamsList is not None:
            seamsList.append(width)

    img = np.stack(eqImgList, axis=1)
    if outImageName is not None:
        UtilArrayToImageFile(img, outImageName)
    return img

def UtilStitchImagesHor(imgInputList, outImageName=None, padMode='constant', constValue=127., seamsList=None):
    imgList = []
    for img in imgInputList:
        if isinstance(img, str):
            img = UtilImageFileToArray(img)
        assert len(img.shape) == 3
        imgList.append(img)

    # Make them all of the same height
    maxHeight = max([x.shape[0] for x in imgList])
    eqImgList = []
    for img in imgList:
        if padMode=='constant':
            img = np.pad(img, ((maxHeight - img.shape[0], 0), (0,0), (0,0)), mode=padMode, constant_values=constValue)
        elif padMode=='reflect':
            img = np.pad(img, ((maxHeight - img.shape[0], 0), (0,0), (0,0)), mode=padMode)
        eqImgList.append(img)

    width = 0
    for img in eqImgList:
        width += img.shape[1]
        if seamsList is not None:
            seamsList.append(width)

    img = np.concatenate(eqImgList, axis=1)
    if outImageName is not None:
        UtilArrayToImageFile(img, outImageName)
    return img


def UtilStitchImagesVert(imgInputList, outImageName=None, padMode='constant', constValue=127., seamsList=None):
    imgList = []
    for img in imgInputList:
        if isinstance(img, str):
            img = UtilImageFileToArray(img)
        assert len(img.shape) == 3
        imgList.append(np.transpose(img, (1,0,2)))

    img = UtilStitchImagesHor(imgList, outImageName=None, padMode=padMode, constValue=constValue, \
                              seamsList=seamsList)
    img = np.transpose(img, (1,0,2))
    if outImageName is not None:
        UtilArrayToImageFile(img, outImageName)
    return img


def UtilImageEdge(img):
    # TODO: let's not use PIL for that
    if isinstance(img, str):
        img = Image.open(img, "r")
    else:
        assert isinstance(img, np.ndarray)
        img = Image.fromarray(img)
    return np.array(img.filter(ImageFilter.FIND_EDGES))


def UtilRemapImage(img, map, fillMethod=None, fillValue=None, ky=3, kx=3):
    """
    Mapping image geometrically
    :param img: input image
    :param map: input map
    :param excludedReflRect: if fillMethod='reflect', then this rectangle should not be mapped by the reflected area
    :return: new mapped image
    """
    h = img.shape[0]
    w = img.shape[1]
    assert map.shape[:2] == (h,w)
    imgArr = img.astype(np.float32)
    if len(imgArr.shape) == 3:
        f = [interpolate.RectBivariateSpline(range(h), range(w), imgArr[:,:,i], ky=ky, kx=kx) for i in range(3)]
        mono = False
    else:
        f = interpolate.RectBivariateSpline(range(h), range(w), imgArr, ky=ky, kx=kx)
        mono = True

    # Deal with the mapped values outside of the source image border
    fillMap = None
    if fillMethod is None:
        pass # No action
    elif (fillMethod == "constant") or (fillMethod == "orig"):
        fillMap = np.logical_and(map[:, :, 0] <= float(h-1), map[:, :, 1] <= float(w-1))
        fillMap = np.logical_and(fillMap, map[:, :, 0] > 0.)
        fillMap = np.logical_and(fillMap, map[:, :, 1] > 0.)
        assert fillMap.shape == (h, w)
    elif fillMethod == "reflect":
        map = UtilReflectCoordTensor(map)
    else:
        raise ValueError('Wrong value of fillMethod: %s' % fillMethod)

    yFlat = np.repeat(np.array(range(h)), w)
    xFlat = np.tile(np.array(range(w)), h)
    remap = map[yFlat, xFlat]
    assert remap.shape == (h * w, 2)
    yImgCoord = remap[:,0]
    xImgCoord = remap[:,1]

    if mono:
        newArr = f(yImgCoord, xImgCoord, grid=False).reshape((h,w))
        if fillMethod == 'constant':
            newArr = np.where(fillMap, newArr, fillValue)
        elif fillMethod == 'orig':
            newArr = np.where(fillMap, newArr, img)
    else:
        newArr = [func(yImgCoord, xImgCoord, grid=False).reshape(h,w) for func in f]
        if fillMethod == 'constant':
            newArr = [np.where(fillMap, arr, fillValue) for arr in newArr]
        elif fillMethod == 'orig':
            newArr = [np.where(fillMap, arr, img[:,:,ind]) for ind,arr in enumerate(newArr)]
        newArr = np.dstack(newArr)

    assert newArr.shape[:2] == (h,w)
    return newArr.clip(min=0., max=255.)


def UtilColorBrightness(color):
    r, g, b = color
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.




def UtilAnyImageRect(img, rect, paddingMode='linear_ramp', **kwargs):
    """
    Returns a rectangle cut out of teh image. If part of this rectangle is o
    :param img:
    :param rect:
    :return:
    """
    h, w = img.shape[:2]
    yMin, xMin, yMax, xMax = rect
    padYBefore = 0 if yMin >= 0 else (-yMin)
    padXBefore = 0 if xMin >=0 else (-xMin)
    padYAfter = 0 if yMax <= h else (yMax - h)
    padXAfter = 0 if xMax <= w else (xMax - w)

    img = np.pad(img, ((padYBefore, padYAfter), (padXBefore, padXAfter)), mode=paddingMode, kwargs=kwargs)
    return img[yMin+padYBefore:yMax+padYBefore, xMin+padXBefore:xMax+padXBefore]


class ImageAnnot(UtilObject):
    """
    Class that wraps image to efficiently add annotations.
    Internally, it uses PIL Image representation
    """

    def __init__(self, img):
        if isinstance(img, str):
            self.name = img
            self.image = Image.open(img, "r")
        else:
            assert isinstance(img, np.ndarray)
            self.name = "ARRAY"
            self.image = Image.fromarray(UtilImageToInt(img), mode="RGB")
        assert self.image.mode == "RGB"
        self.size = self.image.size
        self.clear()

    def clear(self):
        self.xorImage = None
        self.xorName = None
        self.transpImage = None
        self.overImage = Image.new("RGBA", self.size)

    def save(self, outImgName=None):
        image = self.image.copy()
        if self.xorImage:
            assert self.xorImage.mode == "L"
            imgInvert = ImageChops.invert(image)
            image.paste(imgInvert, (0,0), self.xorImage)
        image.paste(self.overImage, (0,0), self.overImage)
        if self.transpImage:
            image = Image.fromarray(np.concatenate([np.asarray(image), \
                np.expand_dims(np.asarray(self.transpImage), axis=2)], axis=2), mode="RGBA")
        if outImgName is not None:
            image.save(outImgName)
        return np.array(image)

    @staticmethod
    def convertToGray(pilImg):
        if (pilImg.mode == "P"):
            pilImg = pilImg.convert(mode="RGB")
        if (pilImg.mode == "RGB") or (pilImg.mode == "RGBA"):
            pilImg = pilImg.convert(mode="L")
        if pilImg.mode != "L":
            raise ValueError("TranspImage is in wrong mode %s" % pilImg.mode)
        return pilImg

    def addAnnotPoint(self, y, x, size = 2, color = (0,0,0)):
        if (x < 0) or (x > self.size[0]-1) or \
                (y < 0) or (y > self.size[1]-1):
            raise ValueError("%s: %d,%d is outside of image size %s" % (self.name,x,y,self.size))
        upperLeftMarg = min(x, y, size)
        upperRightMarg = min(self.size[0]-1-x, y, size)
        lowerRightMarg = min(self.size[0]-1-x, self.size[1]-1-y, size)
        lowerLeftMarg = min(x, self.size[1]-1-y, size)
        upperLeft = (x-upperLeftMarg, y-upperLeftMarg)
        lowerLeft = (x-lowerLeftMarg, y+lowerLeftMarg)
        upperRight = (x+upperRightMarg, y-upperRightMarg)
        lowerRight = (x+lowerRightMarg, y+lowerRightMarg)
        imgDraw = ImageDraw.Draw(self.overImage)
        imgDraw.line([lowerLeft, upperRight], fill=color)
        imgDraw.line([upperLeft, lowerRight], fill=color)

    def addDoubleAnnotPoint(self, y1, x1, y2, x2, size=2, color1=(0,255,0), color2=(255,0,0), colorLine=(0,0,0)):
        self.addAnnotPoint(y1, x1, size, color1)
        self.addAnnotPoint(y2, x2, size, color2)
        imgDraw = ImageDraw.Draw(self.overImage)
        imgDraw.line([(x1,y1), (x2,y2)], fill=colorLine)

    def setXorMask(self, img):
        if isinstance(img, str):
            self.xorName = img
            self.xorImage = Image.open(img, "r")
        else:
            assert isinstance(img, np.ndarray)
            self.xorName = None
            self.xorImage = Image.fromarray(img)
        self.xorImage = self.convertToGray(self.xorImage)

    def setBoundaryMask(self, img, colorExtern=(255,0,0), colorIntern=(0,255,0)):
        """
        Adds internal and external mask border
        :param img: numpy ndarray: greyscale (int or float, both 0-255), or boolean
        :param colorExtern:
        :param colorIntern:
        :return:
        """
        if isinstance(img, str):
            img = UtilImageFileToArray(img)
        else:
            assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2
        if img.dtype != np.bool:
            img = (img >= 128)
        boundaries = UtilRegionBoundaries(img)
        externalCoord = np.transpose(np.logical_and(boundaries, np.logical_not(img)).nonzero())
        externalCoord = np.flip(externalCoord, axis=1).flatten() # PIL requires x,y
        internalCoord = np.transpose(np.logical_and(boundaries, img).nonzero())
        internalCoord = np.flip(internalCoord, axis=1).flatten()
        imgDraw = ImageDraw.Draw(self.overImage)
        imgDraw.point(list(externalCoord), fill=colorExtern)
        imgDraw.point(list(internalCoord), fill=colorIntern)

    def setTransparencyMask(self, img, binarizeThreshold=None):
        if isinstance(img, str):
            self.transpName = img
            self.transpImage = Image.open(img, "r")
        else:
            assert isinstance(img, np.ndarray)
            self.transpName = None
            self.transpImage = Image.fromarray(img)
        self.transpImage = self.convertToGray(self.transpImage)
        if binarizeThreshold is not None:
            self.transpImage = self.transpImage.point(lambda p: p > binarizeThreshold and 255)
        return np.array(self.transpImage)


class ImageAnnotPlot(UtilObject):

    def __init__(self, img, totalHeight=None, totalWidth=None):
        assert (totalHeight is not None) or (totalWidth is not None)
        if totalHeight is not None:
            totalWidth = totalHeight * img.shape[1] // img.shape[0]
        else:
            totalHeight = totalWidth * img.shape[0] // img.shape[1]
        self.height = totalHeight
        self.width = totalWidth
        plt.figure(figsize=(self.width/100, self.height/100), dpi=100)
        img = UtilImageToInt(img)
        plt.imshow(img)
        plt.xlim((0, img.shape[1]))
        plt.ylim((img.shape[0], 0))

    @staticmethod
    def scaleColor(color):
        return tuple(x / 255. for x in color)

    @staticmethod
    def pointPairToLineCoord(p1, p2):
        return (np.array([p1[1], p2[1]]), np.array([p1[0], p2[0]]))

    def addConnectedPoints(self, points, connections, color=(255, 0, 0)):
        """
        :param points: point coordinates, shape (N, 2), type float
        :param connections: connections in terms of points, shape (K, 2), type uint
        :param color: (R, G, B) 0 - 255
        :return:
        """

        color = self.scaleColor(color)
        for p in points:
            plt.plot(p[1], p[0], 'o', color=color)
        pointPairs = points[connections]
        for pp in pointPairs:
            self.pointPairToLineCoord(pp[0], pp[1])
            plt.plot(*self.pointPairToLineCoord(pp[0], pp[1]), color=color, linestyle='-', linewidth=1)
        for ind, p in enumerate(points):
            plt.text(p[1], p[0], ' ' + str(ind + 1), color=color, fontsize=self.height / 100)

    def addConnectedAndSecondaryPoints(self, points, secPoints, connections, color=(255, 0, 0), secColor=(0, 255, 0)):
        assert points.shape[0] == secPoints.shape[0]
        self.addConnectedPoints(points, connections, color)
        secColor = self.scaleColor(secColor)
        for p in secPoints:
            plt.plot(p[1], p[0], 'o', color=secColor)
        for ind in range(points.shape[0]):
            plt.plot(*self.pointPairToLineCoord(points[ind], secPoints[ind]),
                     color=secColor, linestyle='-', linewidth=1)

    def save(self, fileName):
        plt.savefig(fileName, bbox_inches='tight')
        plt.close()


class BoundingBox(UtilObject):
    def __init__(self, image=None):
        self._rect = np.array([forwardCompat.maxint, forwardCompat.maxint, -1, -1])
        if image is not None:
            self.addMask(image)

    def addMask(self, image):
        if isinstance(image, str):
            image = UtilImageFileToArray()
        if len(image.shape) == 3:
            image = UtilFromRgbToGray(image)
        if image.dtype != np.bool:
            image = image > 128
        ysWithPoints = np.any(image, axis=1)
        xsWithPoints = np.any(image, axis=0)
        if not (np.any(ysWithPoints) or np.any(xsWithPoints)):
            return
        yMin, yMax = np.where(ysWithPoints)[0][[0, -1]] + [0,1]
        xMin, xMax = np.where(xsWithPoints)[0][[0, -1]] + [0,1]
        self.accomodate([yMin, xMin, yMax, xMax])

    def addPoint(self, yCoord, xCoord):
        self.accomodate([yCoord, xCoord, yCoord+1, xCoord+1])

    def addMultiPoint(self, array):
        assert (len(array.shape) == 2) and (array.shape[1] == 2)
        for p in array:
            self.addPoint(int(round(p[0])), int(round(p[1])))

    def accomodate(self, updateRect):
         updateRect = np.array(updateRect).astype(np.int)
         self._rect[:2] = np.where(self._rect[:2] > updateRect[:2], updateRect[:2], self._rect[:2])
         self._rect[2:] = np.where(self._rect[2:] < updateRect[2:], updateRect[2:], self._rect[2:])

    @property
    def rect(self):
        return tuple(self._rect)

class BoundingBoxStats(UtilObject):
    """
    Calculates statistics of the object position in the picture
    """
    def __init__(self):
        self.bBoxes = []

    def addBBox(self, shape, bb):
        self.bBoxes.append((shape[0], shape[1]) + bb)

    def stats(self):
        aspRatio = []
        centerDistY = []
        centerDistX = []
        relHeight = []
        relWidth = []
        for h, w, yMin, xMin, yMax, xMax in self.bBoxes:
            yCenter = (yMax + yMin) / 2
            xCenter = (xMax + xMin) / 2
            try:
                aspRatio.append((yMax - yMin) / (xMax - xMin))
                centerDistY.append(2 * (yCenter - h/2) / (h - (yMax - yMin)))
                centerDistX.append(2 * (xCenter - w/2) / (w - (xMax - xMin)))
                relHeight.append((yMax - yMin) / h)
                relWidth.append((xMax - xMin) / w)
            except (ZeroDivisionError, FloatingPointError):
                continue

        d = {}
        for name in ['aspRatio', 'centerDistY', 'centerDistX', 'relHeight', 'relWidth']:
            d[name] = self.getStat(name, locals()[name])
        return d

    def getStat(self, name, statList):
        return (np.mean(statList), np.std(statList))







