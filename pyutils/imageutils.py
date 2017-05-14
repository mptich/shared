# Image utilities

__author__ = "Misha Orel"

import shared.pyutils.forwardCompat as forwardCompat
from shared.pyutils.utils import *
from shared.pyutils.tensorutils import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageFilter
import scipy.ndimage.filters as scipyFilters
from scipy import interpolate
import scipy
import math
import sys
from sklearn import linear_model
import gc
import time
import operator
import collections
import csv
import cv2

scipyVerMaj, scipyVerMin, _ = scipy.__version__.split(".")
if (int(scipyVerMaj) == 0) and (int(scipyVerMin) < 14):
    raise Exception("SCIPY version %s, should be at least 0.14.0" % scipy.__version__)

def UtilImageFileToArray(fileName):
    img = cv2.imread(fileName)
    if (len(img.shape) == 3) and (img.shape[2] == 3):
        img = np.flip(img, axis=2)
    return UtilImageToFloat(img)

def UtilArrayToImageFile(arr, fileName):
    arr = UtilImageToInt(arr)
    if (len(arr.shape) == 3) and (arr.shape[2] == 3):
        arr = np.flip(arr, axis=2)
    cv2.imwrite(fileName, arr)

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

def UtilValidateBoundBox(shape, bb, margin=0):
    h,w=shape
    yMin,xMin,yMax,xMax = bb
    return ((yMin >= margin) and (yMin < h-margin) and (xMin >= margin) and (xMin < w-margin) and \
            (yMax <= h-margin) and (xMax <= w-margin) and (yMin < yMax) and (xMin < xMax))


def UtilVerticalScale(img, destHeight, destWidth, padValue = 0., applyGauss=False):
    """
    Scales image in such a way that it is fit by height, and width either cropped or padded
    :param img: input image
    :param destHeight: desired height
    :param destWidth: desired width
    :param padValue: in case of padding, what value to use
    :return: tuple (Scaled image, scale ratio, left pad)
    """

    h,w = img.shape[:2]

    # Scale by height
    scale = destHeight / h
    realWidth = int(round(w * scale))
    img = UtilImageResize(img, destHeight, realWidth, applyGauss=applyGauss)

    # See if padding / clipping is needed
    leftPadWidth = (destWidth - realWidth) // 2
    rightPadWidth = destWidth - realWidth - leftPadWidth
    assert (leftPadWidth * rightPadWidth) >= 0
    if (leftPadWidth + rightPadWidth) > 0:
        padTuple = ((0, 0), (leftPadWidth, rightPadWidth))
        if len(img.shape) == 3:
            padTuple += ((0,0),)
        img = np.pad(img, padTuple, mode='constant', constant_values=padValue)
    else:
        img = img[:, -leftPadWidth:realWidth + rightPadWidth]

    assert img.shape[:2] == (destHeight, destWidth)
    return (img, scale, leftPadWidth)


def UtilImageResize(img, destHeight, destWidth, applyGauss=True):
    """
    It uses PIL algorithms. So we have to use gaussian
    :param img: input image
    :param destHeight: target height
    :param destWidth: target width
    :return:
    """
    h, w = img.shape[:2]
    depth = None
    if len(img.shape) == 3:
        depth = img.shape[2]
    def gaussianCalc(srcSize, destSize):
        if destSize >= srcSize:
            return 0.2
        return 0.4 * srcSize / destSize
    ySigma = gaussianCalc(h,destHeight)
    xSigma = gaussianCalc(w,destWidth)
    if applyGauss:
        if depth is None:
            img = scipyFilters.gaussian_filter(img, (ySigma, xSigma))
        else:
            img = np.dstack([scipyFilters.gaussian_filter(img[:,:,i], (ySigma, xSigma)) for i in range(depth)])
    img = scipy.misc.imresize(img, (destHeight, destWidth))
    return img

def UtilStitchImagesHor(imgNameList, outImageName):
    imgList = []
    for name in imgNameList:
        imgList.append(Image.open(name))

    width = 0
    height = 0
    for image in imgList:
        w, h = image.size
        width += w
        if height < h:
            height = h

    start = 0
    result = Image.new('RGB', (width, height))
    for image in imgList:
        result.paste(im=image, box=(start, 0))
        start += image.size[0]

    result.save(outImageName)

def UtilImageEdge(img):
    # TODO: let's not use PIL for that
    if isinstance(img, str):
        img = Image.open(img, "r")
    else:
        assert isinstance(img, np.ndarray)
        img = Image.fromarray(img)
    return np.array(img.filter(ImageFilter.FIND_EDGES))


def UtilRemapImage(img, map, fillMethod="constant", fillValue=127., ky=3, kx=3):
    """
    Mapping image geometrically
    :param img: input image
    :param map: input map
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
    if fillMethod == "constant":
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
        if fillMap is not None:
            newArr = np.where(fillMap, newArr, fillValue)
    else:
        newArr = [func(yImgCoord, xImgCoord, grid=False).reshape(h,w) for func in f]
        if fillMap is not None:
            newArr = [np.where(fillMap, arr, fillValue) for arr in newArr]
        newArr = np.dstack(newArr)

    assert newArr.shape[:2] == (h,w)
    return newArr.clip(min=0., max=255.)


def UtilDbgMatrixToImage(mat, imageName = None, method ="direct"):
    """
    Mostly for debug purposes: convert "arbitrary" marix to an image array
    :param mat: input matrix
    :param imageName: file to save in, optional
    :param method: "direct", "flat_hist"
    :return:
    """
    shape = np.shape(mat)
    if (len(shape) > 3) or (len(shape) < 2):
        raise ValueError("UtilDbgMatrixToImage wrong shape %s" % repr(shape))
    if len(shape) == 2:
        count = 1
    else:
        count = shape[2]
        if count == 1:
            mat = np.reshape(mat, shape[:2])
    if (count > 3) or (count == 0):
        raise ValueError("UtilDbgMatrixToImage wrong last dimension %d" % count)

    if count == 2:
        mat = np.dstack([mat[:,:,0], mat[:,:,1], np.zeros(shape[:2], dtype = np.float32)])
        count = 3

    if method == "direct":
        temp = mat if count != 1 else mat.reshape(np.shape(mat)+(1,))
        maxVal, minVal = (np.amax(temp, axis=(0,1)), np.amin(temp, axis=(0,1)))
        diff = (maxVal - minVal).clip(min=UtilNumpyClippingValue(np.float32))
        if count == 1:
            img = (mat - minVal[0])* 255.0 / diff[0]
        elif count == 3:
            img = np.multiply((mat - minVal) * 255.0, np.reciprocal(diff))
        else:
            raise ValueError("No implemented")
    elif method == "flat_hist":
        if count == 1:
            l = sorted(mat.flatten())
            length = len(l)
            lBound = []
            for i in range(255):
                lBound.append(l[length * i // 255])
            img = np.searchsorted(lBound, mat)
        elif (count == 3):
            images = [UtilDbgMatrixToImage(mat[:, :, i], method=method) for i in range(count)]
            img = np.dstack([images[i] for i in range(count)])
        else:
            raise ValueError("No implemented")
    else:
        raise ValueError("No implemented")

    img = img.clip(min=0., max=255.)

    if imageName is not None:
        if len(img.shape) == 2:
            img = UtilFromGrayToRgb(img)
        UtilArrayToImageFile(img, imageName)

    return img



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
            self.image = Image.fromarray(img, mode="RGB")
        assert self.image.mode == "RGB"
        self.size = self.image.size
        self.clear()

    def clear(self):
        self.xorImage = None
        self.xorName = None
        self.overImage = None
        self.transpImage = None

    def save(self, outImgName=None):
        image = self.image.copy()
        if self.xorImage:
            assert self.xorImage.mode == "L"
            imgInvert = ImageChops.invert(image)
            image.paste(imgInvert, (0,0), self.xorImage)
        if self.overImage:
            assert self.overImage.mode == "RGBA"
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

    def addAnnotPoint(self, x, y, size = 2, color = (0,0,0)):
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
        if self.overImage is None:
            self.overImage = Image.new("RGBA", self.size)
        imgDraw = ImageDraw.Draw(self.overImage)
        imgDraw.line([lowerLeft, upperRight], fill=color)
        imgDraw.line([upperLeft, lowerRight], fill=color)

    def addDoubleAnnotPoint(self, x1, y1, x2, y2, size=2, color1=(0,255,0), color2=(255,0,0), colorLine=(0,0,0)):
        self.addAnnotPoint(x1, y1, size, color1)
        self.addAnnotPoint(x2, y2, size, color2)
        assert self.overImage is not None
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



class CVImage(UtilObject):
    """
    Image in the format convenient for CV processing
    """

    def __init__(self, image):
        self.cleanup()
        if isinstance(image, CVImage):
            data = np.copy(image.data)
        else:
            assert isinstance(image, np.ndarray)
            data = image

        # Normalize
        meanVal = np.mean(data, axis=(0,1))
        stdVal = np.std(data, axis = (0,1)).clip(min=UtilNumpyClippingValue(np.float32))
        self.data = np.multiply(data - meanVal, np.reciprocal(stdVal))

    def cleanup(self):
        self.fc = None
        self.fcFine = None
        self.fcNorm = None
        self.fcFineNorm = None
        self.fine = None
        self.meanCell = None
        self.neibStd = None
        self.neibCells = None
        self.neibCellsFine = None

    def quickSubsample(self):
        data = self.data
        h, w, _ = np.shape(self.data)
        hend = h if (h & 1) == 0 else h-1
        wend = w if (w & 1) == 0 else w-1
        self.data = (data[0:hend:2,0:wend:2,:]+data[0:hend:2,1::2,:]+data[1::2,0:wend:2,:]+data[1::2,1::2,:]) / 4.0

    def subsample(self, times):
        h, w, _ = np.shape(self.data)
        img = self.image()
        newSize = (int(h/times), int(w/times)) # This order for scipy
        data = scipy.misc.imresize(img, newSize, interp="lanczos", mode="RGB")
        self.__init__(image=data)

    def gaussian(self, sigma):
        self.data = np.dstack([scipyFilters.gaussian_filter(self.data[:,:,i], sigma=sigma) for i in range(3)])

    def image(self, imageName=None):
        return UtilDbgMatrixToImage(self.data, imageName=imageName, method ="direct")

    @staticmethod
    def weightRing(radius):
        """
        Returns tuple (lc, ld), where lc is a list of tuples (x,y), and ld is a list of weights, for all pixels
        within radius
        """
        lc = []
        ld = []
        r = int(radius)
        rsq = radius * radius
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                dsq = float(i*i + j*j)
                if rsq >= dsq:
                    lc.append((i, j))
                    #ld.append(rsq / (rsq + dsq))
                    ld.append(1.)
        return (lc, ld)

    @staticmethod
    def linearFeat(x, y):
        return [1., y, x]

    @staticmethod
    def squareFeat(x, y):
        return [1., y, x, y*y, x*y, x*x]

    @staticmethod
    def cubicFeat(x, y):
        return [1., y, x, y*y, x*y, x*x, y*y*y, x*y*y, x*x*y, x*x*x]

    @staticmethod
    def quadFeat(x, y):
        return [1., y, x, y*y, x*y, x*x, y*y*y, x*y*y, x*x*y, x*x*x, y*y*y*y, x*y*y*y, x*x*y*y, x*x*x*y, x*x*x*x]

    def cells(self, radius, featFunc):
        self.cleanup()
        radint = int(radius)
        wring = self.weightRing(radius)
        weights = wring[1]
        nSamples = len(weights)
        nFeatures = len(featFunc(0., 0.))
        features = np.ndarray((nSamples, nFeatures), dtype=np.float32)
        featureCoord = np.ndarray((nSamples, 2), dtype=np.float32)
        for i, e in enumerate(wring[0]):
            x, y = (float(v) for v in e)
            features[i,:] = featFunc(x, y)
            featureCoord[i] = [x,y]
        self.featFunc = featFunc
        self.nFeatures = nFeatures
        self.nSamples = nSamples
        self.weights = weights
        self.features = features
        self.featureCoord = featureCoord
        self.radius = radius
        self.radint = radint

        print ("Initial nSamples %s" % nSamples)

        baseShape = np.shape(self.data)[0:2]
        viewWidth = baseShape[1] - 2 * radint
        viewHeight = baseShape[0] - 2 * radint
        viewHorEnd = baseShape[1] - radint
        viewVertEnd = baseShape[0] - radint
        self.viewWidth = viewWidth
        self.viewHeight = viewHeight
        self.viewHorEnd = viewHorEnd
        self.viewVertEnd = viewVertEnd

        # Collect colors of the neighbouring cells
        neibCells = np.stack([self.data[radint+y:viewVertEnd+y,radint+x:viewHorEnd+x] for x,y in wring[0]], axis=3)
        self.meanCell, self.neibStd, self.neibCells = self.normalizeCells(neibCells)

    @staticmethod
    def correlate(cells1, cells2, upperLeft, lowerRight, std1=None, std2=None):
        """
        :param cells1, cells2: 4D arrays (height,width,color,values)
        :param upperLeft: (dx, dy) upper left corner of movement
        :param lowerRight: (dx, dy) lower right corner of movement
        :param std1, std2: if present, show per color STD of the original array
        :return: 2 4-D array, showing correlations, and weights of these correlations,
            in terms of (dx, dy), for each pixel. Weights array migt be None (then all correlations are of
            equal weight)
        """

        viewHeight, viewWidth, nColors, nSamples = np.shape(cells1)
        viewHeightOther, viewWidthOther, nColorsOther, nSamplesOther = np.shape(cells2)
        gridRatio = viewHeightOther / viewHeight
        if (viewHeight * gridRatio != viewHeightOther) or (viewWidth * gridRatio != viewWidthOther):
            raise ValueError("Wronf axes 0/1 in correlate: (%d, %d) vs (%d, %d)" % \
                             (viewHeight, viewWidth, viewHeightOther, viewWidthOther))
        if (nColors != 3) or (nColorsOther != 3):
            raise ValueError("In correlate: wrong number of colors: %d %d" % (nColors, nColorsOther))
        if (nSamples != nSamplesOther):
            raise ValueError("In correlate: nSamples are different: %d %d" % (nSamples, nSamplesOther))

        minX, minY = upperLeft
        maxX, maxY = lowerRight
        startHorView = -minX if minX < 0 else 0
        startOtherHor = minX if minX > 0 else 0
        endHorView = viewWidth - maxX if maxX > 0 else viewWidth
        startVertView = -minY if minY < 0 else 0
        startOtherVert = minY if minY > 0 else 0
        endVertView = viewHeight - maxY if maxY > 0 else viewHeight
        if (startHorView >= endHorView) or (startVertView >= endVertView):
            raise ValueError("correlateFine: incorrect coordinates: %s %s" % (repr(upperLeft), repr(lowerRight)))
        minXFine,maxXFine,minYFine,maxYFine = (gridRatio*x for x in (minX,maxX,minY,maxY))
        startHorViewFine,startOtherHorFine,endHorViewFine,startVertViewFine,startOtherVertFine,endVertViewFine = \
            (gridRatio*x for x in (startHorView,startOtherHor,endHorView,startVertView,startOtherVert,endVertView))

        tensor1 = cells1[startVertView:endVertView, startHorView:endHorView]
        output = np.empty((endVertView-startVertView, endHorView-startHorView, maxYFine-minYFine, maxXFine-minXFine))

        for j in range(maxYFine-minYFine):
            for i in range(maxXFine-minXFine):
                tensor2 = cells2[ \
                    startOtherVertFine+j:startOtherVertFine+j+endVertViewFine-startVertViewFine:gridRatio, \
                    startOtherHorFine+i:startOtherHorFine+i+endHorViewFine-startHorViewFine:gridRatio]
                diff = tensor2 - tensor1
                output[:,:,j,i] = np.sum(diff * diff, axis=(2,3))

        weight = None
        if (std1 is not None) and (std2 is not None):
            weight1 = np.prod(std1, axis=2)
            weight2 = np.prod(std2, axis=2)
            weight = np.where(weight1 < weight2, x=weight1, y=weight2)

        return (output, weight)

    def localMinima(self, corr, weight=None):
        """
        :param corr: correlation matrix returned from correlate()
        :param weight: weight returned from correlate
        :return: 4-D ndarray containing True for local minimas
        """

        # First find the cut-off level. Let's have it at 10% of correlation values
        count = 1000
        temp = np.random.choice(corr.flatten(), count)
        temp.sort()
        corrLocalMinimaThreshold = temp[count / 10]
        localMinima = np.logical_and( \
            corr[:,:,1:-1,1:-1] < corr[:,:,:-2,1:-1], corr[:,:,1:-1,1:-1] < corr[:,:,1:-1,:-2])
        localMinima = np.logical_and(localMinima, corr[:,:,1:-1,1:-1] < corr[:,:,2:,1:-1])
        localMinima = np.logical_and(localMinima, corr[:,:,1:-1,1:-1] < corr[:,:,1:-1,2:])
        localMinima = np.logical_and(localMinima, corr[:,:,1:-1,1:-1] < corrLocalMinimaThreshold)
        return localMinima

    def fineCells(self, fine):
        """
        Creates a fine structure out of neibCells, by extrapolating via a polinomial
        :param fine: fine=1 yields 3x fine structure, fine=2 - 5x, fine=3 - 7x, etc
        :return:
        """
        self.fine = fine
        # First, approximate neibCells with a polinomial
        nTargets = self.viewWidth * self.viewHeight * 3

        neibCells = self.neibCells
        labels = np.transpose(neibCells, axes=(3,0,1,2)).reshape(self.nSamples, nTargets)

        reg = linear_model.LinearRegression(fit_intercept=False, n_jobs=-1)
        reg.fit(self.features, labels, sample_weight=self.weights)

        fc = reg.coef_.reshape((self.viewHeight, self.viewWidth, 3, self.nFeatures))
        self.fc = fc

        # Now create the fine structure of neighbour cells
        gridRatio = 1 + 2 * fine
        smallRadius = self.radius - 1. # Take subset of the samples
        smallFeatCoord = np.array([e for e in self.featureCoord if (e[0]*e[0]+e[1]*e[1]) <= smallRadius*smallRadius])
        sampleCount = len(smallFeatCoord)
        neibCellsFine = np.empty((self.viewHeight * gridRatio, self.viewWidth * gridRatio, 3, sampleCount))
        for i in range(-fine,fine+1):
            for j in range(-fine,fine+1):
                xOff = float(i) / gridRatio
                yOff = float(j) / gridRatio
                features = np.array([self.featFunc(x+xOff,y+yOff) for x,y in smallFeatCoord])
                neibCellsFine[fine+j::gridRatio, fine+i::gridRatio] = np.tensordot(fc, features, axes = [[3],[1]])
        self.meanCellFine, self.neibStdFine, self.neibCellsFine = self.normalizeCells(neibCellsFine)

    @staticmethod
    def normalizeCells(cells):
        """
        :param cells: 4D array of cells (height,width,color,values)
        :return: tuple (meanCells(height,width,color), std(height,width,color), normilizedCells(height,width,
                color,values)
        """
        meanCell = np.mean(cells, axis=(3,))
        diffCells = cells - meanCell.reshape(np.shape(meanCell) + (1,))
        std = np.std(diffCells, axis=(3,)).clip(min=UtilNumpyClippingValue(np.float32))
        cells = np.multiply(diffCells, np.reciprocal(std).reshape((np.shape(std) + (1,))))
        return (meanCell, std, cells)


    @staticmethod
    def neibCells2D(neibCells, featureCoord):
        """
        :param neibCells: array of neibCells, size of self.nSamples
        :return: neibCells as 2D array; pixels not belonging to it a 0s
        """
        radint = int(np.max(featureCoord))
        size = radint * 2 + 2 # extra 1 for the black border
        if len(np.shape(neibCells)) < 3:
            # If the 2st 2 dimensions are missing, amke them 1 "pixel"
            neibCells = neibCells.reshape((1,1)+np.shape(neibCells))
        if len(np.shape(neibCells)) < 4:
            # If colors are missing
            x,y,z = np.shape(neibCells)
            neibCells = np.repeat(neibCells.reshape((x,y,1,z)), repeats=3, axis=2)
        neibCells = np.transpose(neibCells, axes=(0,1,3,2))
        shape = np.shape(neibCells)
        nSamples = len(featureCoord)
        assert nSamples == shape[2]
        minVal = np.min(neibCells)
        maxVal = np.max(neibCells)
        missingVal = minVal - 0.3 * (maxVal-minVal)
        ret = np.empty((shape[0], size, shape[1], size, 3), dtype=np.float32)
        ret.fill(missingVal)
        for k in range(nSamples):
            i,j = (int(x) for x in featureCoord[k])
            ret[:,radint+j,:,radint+i,:] = neibCells[:,:,k,:]
        ret = ret.reshape((shape[0]*size, shape[1], size, 3)).reshape((shape[0]*size, shape[1]*size, 3))
        return ret

    def edge(self):
        """
        Use Sobel filters
        :return: matrix (height, width) of detected edges
        """
        horFilt = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        vertFilt = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
        horEdges = np.dstack([scipyFilters.convolve(self.data[:,:,i], horFilt) for i in range(3)])
        vertEdges = np.dstack([scipyFilters.convolve(self.data[:,:,i], vertFilt) for i in range(3)])
        horEdges = np.sum(horEdges * horEdges, axis=2)
        vertEdges = np.sum(vertEdges * vertEdges, axis=2)
        self.edges = np.sqrt(horEdges + vertEdges)
        return UtilDbgMatrixToImage(self.edges)

    def meanSharpness(self):
        h,w = self.edges.shape
        return np.sum(self.edges) / (h*w)


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





