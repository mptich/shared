# Image utilities

import shared.pyutils.forwardCompat
from shared.pyutils.utils import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageFilter
import csv

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
    if isinstance(img, str):
        img = Image.open(img, "r")
    return img.filter(ImageFilter.FIND_EDGES)



class ImageAnnot(UtilObject):
    """
    Class that wraps image to efficiently add annotations
    """

    def __init__(self, img):
        if isinstance(img, str):
            self.name = img
            self.image = Image.open(img, "r")
        else:
            self.name = None
            self.image = img
        assert self.image.mode == "RGB"
        self.size = self.image.size
        self.clear()

    def clear(self):
        self.xorImage = None
        self.xorName = None
        self.overImage = None

    def save(self, outImgName):
        image = self.image.copy()
        if self.xorImage:
            assert self.xorImage.mode == "L"
            imgInvert = ImageChops.invert(image)
            image.paste(imgInvert, (0,0), self.xorImage)
        if self.overImage:
            assert self.overImage.mode == "RGBA"
            image.paste(self.overImage, (0,0), self.overImage)
        image.save(outImgName)

    def addAnnotPoint(self, x, y, size = 1, color = (0,0,0)):
        if (x < 0) or (x > self.size[0]-1) or \
                (y < 0) or (y > self.size[1]-1):
            raise ValueError("%d,%d is outside of image size %s" % (x,y,self.size))
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

    def setXorMask(self, img):
        if isinstance(img, str):
            self.xorName = img
            self.xorImage = Image.open(img, "r")
        else:
            self.xorName = None
            self.xorImage = img
        if self.xorImage.mode == "RGB":
            self.xorImage = self.xorImage.convert(mode="L")
        if self.xorImage.mode != "L":
            raise ValueError("XorImage is in wrong mode %s" % self.xorImage.mode)


