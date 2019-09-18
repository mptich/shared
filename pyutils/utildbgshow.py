# Utilities to dump numpy arrays, show object annotations, etc, for debugging.
#
# Copyright (C) 2017-2018  Author: Misha Orel
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

try:
  from shared.pyutils.imageutils import *
except:
  print("Module imageutils not found")

import numpy as np
import collections
import random

from shared.pyutils.tensorutils import UtilCartesianMatrix


def UtilGetDistinctColors(colorCount):
  # TODO Improve algo
  if colorCount > 12:
    return None
  colorMat = UtilCartesianMatrix([0, 255], [0, 123, 255], [0, 255]).reshape((-1, 3))
  colorBright = [UtilColorBrightness(x) for x in colorMat]
  inds = np.argsort(colorBright)
  return colorMat[inds, :]


def UtilDisplayColorPanel(colorMat, fileName):
  colorMat = colorMat.reshape((1,) + colorMat.shape)
  colorMat = np.repeat(colorMat, 30, axis=1)
  colorMat = np.repeat(colorMat, 30, axis=0)
  UtilArrayToImageFile(colorMat, fileName)


def UtilDbgMatrixToImage(mat, imageName=None, method="direct", **kwargs):
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
  if count == 4:
    # Remove alpha channel
    mat = mat[:, :, :3]
    count = 3
  if count not in (1, 3):
    raise ValueError("UtilDbgMatrixToImage wrong last dimension %d" % count)

  if count == 2:
    mat = np.dstack([mat[:, :, 0], mat[:, :, 1], np.zeros(shape[:2], dtype=np.float32)])
    count = 3

  normalizeByHeight = kwargs.get('normalizeByHeight', None)
  maxVal = kwargs.get('maxVal', None)
  minVal = kwargs.get('minVal', None)
  if normalizeByHeight is None:
    byAxis = (0, 1)
  else:
    byAxis = (1,)
  if maxVal is None:
    maxVal = np.amax(mat, axis=byAxis)
  if minVal is None:
    minVal = np.amin(mat, axis=byAxis)
  if normalizeByHeight is not None:
    if count == 1:
      maxVal = maxVal.reshape(maxVal.shape + (1,))
      minVal = minVal.reshape(minVal.shape + (1,))
    elif count == 3:
      maxVal = maxVal.reshape(maxVal.shape[0] + (1, 3))
      minVal = minVal.reshape(minVal.shape + (1, 3))

  if method == "direct":
    diff = (maxVal - minVal).clip(min=UtilNumpyClippingValue(np.float32))
    img = (mat - minVal) * 255.0 / diff

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

  elif method == "log":
    colorCount = 12
    maxColorIndex = colorCount - 1
    assert count == 1
    if normalizeByHeight is not None:
      if maxVal is None:
        maxVal = np.max(mat, axis=1)
      if len(maxVal.shape) == 1:
        maxVal = maxVal.reshape((-1, 1))
      mat = mat / maxVal
      maxVal = 1.
    base = kwargs.get('base', 2.)
    offMax = kwargs.get('offMax', base)
    if maxVal is None:
      maxVal = np.max(mat)
    maxVal /= offMax
    img = mat.clip(min=UtilNumpyClippingValue(np.float32))
    img = (np.log(img / float(maxVal)) / np.log(base) + maxColorIndex).astype(np.int).clip(min=0,
                                                                                           max=maxColorIndex)
    colorMat = UtilGetDistinctColors(colorCount)
    img = colorMat[img]
  else:
    raise ValueError("No implemented")

  img = img.clip(min=0., max=255.).astype(np.float32)

  if imageName is not None:
    if len(img.shape) == 2:
      img = UtilFromGrayToRgb(img)
    UtilArrayToImageFile(img, imageName)

  return img


def _scaleBrightnessRange(img, interval, axis=None):
  if interval is None:
    if axis is None:
      minVal = np.min(img)
      maxVal = np.max(img)
    else:
      minVal = np.min(img, axis=axis)
      maxVal = np.max(img, axis=axis)
  else:
    if isinstance(interval, tuple):
      minVal, maxVal = interval
    else:
      maxVal = interval
      # Have to cast to type(maxVal) because scalar and shape () array are not the same :)
      minVal = type(maxVal)(np.zeros(maxVal.shape, maxVal.dtype))

  def _reformatClippers(clipVal):
    if np.isscalar(clipVal):
      clipVal = np.array([[clipVal]])
    elif axis == 0:
      clipVal = clipVal.reshape((1, img.shape[1]))
    else:
      assert axis == 1
      clipVal = clipVal.reshape((img.shape[0], 1))
    return clipVal

  minVal = _reformatClippers(minVal)
  maxVal = _reformatClippers(maxVal)
  diff = (maxVal - minVal).clip(min=UtilNumpyClippingValue(img.dtype))
  recipDiff = np.reciprocal(diff)
  img = img.clip(min=minVal, max=maxVal)
  img = (img - minVal) * recipDiff
  return img


def UtilDbg2GrayscalesToImage(gImg, rImg, gInterval=None, rInterval=None, axis=0, fileName=None):
  """
  Combines 2 greyscale iamges into one (green and red colors)
  :param gImg: green image
  :param rImg: red image
  :param ginterval: tuple min and max values of green image (might be ndarray by axis)
  :param rInterval: tuple min and max values of red image (might be ndarray by axis)
  :param fileName:                                                           width = img.shape[1]
  :return:                                                                   # Rectangle might wrap around
  """

  assert gImg.shape == rImg.shape
  assert len(gImg.shape) == 2

  gImg = _scaleBrightnessRange(gImg, gInterval, axis=axis)
  rImg = _scaleBrightnessRange(rImg, rInterval, axis=axis)
  bImg = np.ndarray(shape=gImg.shape, dtype=gImg.dtype)
  bImg.fill(1.)
  img = np.stack([rImg, gImg, bImg], axis=2)
  assert np.min(img) >= 0.
  assert np.max(img) <= 1.
  img = (img * 255.).astype(np.uint8).clip(min=0, max=255)
  if fileName is not None:
    UtilArrayToImageFile(img, fileName)
  return img


def UtilDbg1GrayscaleToImage(img, interval=None, axis=0, fileName=None):
  return UtilDbg2GrayscalesToImage(gImg=img, rImg=img, gInterval=interval, rInterval=interval,
                                   axis=axis, fileName=fileName)


def UtilDbgDisplayAsPolar(img, interval, axis=None, fileName=None, dynamicRangeFunc=None):
  """
  Converts img in cartesian coordinates to polar, displays it
  Phase is shown with color: postive angl
  :param img: Input image in cartesian coordinates (shape (..., 2))
  :param interval: None, or just maxValue, or tuple (minValue, maxValue)
  :param axis: axis ofr usage of min/max values
  :param fileName:
  :param dynamicRangeFunc:
  :return:
  """
  img = UtilCartesianToPolar(img)
  if dynamicRangeFunc is not None:
    img[:, :, 0] = dynamicRangeFunc(img[:, :, 0])
    if interval is not None:
      if isinstance(interval, tuple):
        interval = tuple((dynamicRangeFunc(x) for x in interval))
      else:
        interval = dynamicRangeFunc(interval)
  brt = _scaleBrightnessRange(img[:, :, 0], interval, axis=axis)
  return UtilDbgDisplayPolar(np.stack([brt, img[:, :, 1]], axis=2), fileName=fileName)


def UtilDbgDisplayPolar(polarImg, fileName=None):
  """
  Displays image in polar coordinates
  :param img: numpy array[M, N, 2], where 0 is amplitude [0., 1.], 1 is phase [-pi, pi]
  :return:
  """

  # Start with HLS image
  lumin = (polarImg[:, :, 0] * 128.).astype(np.uint8)
  halfCirc = polarImg[:, :, 1] / np.pi
  halfCirc = np.where(halfCirc >= 0., halfCirc, 2. + halfCirc)
  hue = (halfCirc * 90.).astype(np.uint8)
  satur = np.empty(polarImg.shape[:2], dtype=np.uint8)
  satur.fill(255)
  img = np.stack([hue, lumin, satur], axis=2)
  img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)

  if fileName is not None:
    UtilArrayToImageFile(img, fileName)
  return img


def UtilDrawHistogram(inputList=None, bins='fd', show=True, saveFile=None, logCounts=False):
  if inputList is None:
    if show:
      plt.show()
    return

  hist, binEdges = np.histogram(inputList, bins=bins)
  if logCounts:
    hist = np.log(np.array(hist) + 1)
  plt.plot(binEdges[:-1], hist)
  if show:
    plt.show()
  if saveFile is not None:
    plt.savefig(saveFile)


def UtilDrawBoundingBox(img, yMin, xMin, yMax, xMax, color=(255, 0, 0), density=(1,1)):
  color = np.array(color)
  img = np.copy(img)

  yx_set =  UtilGetBoundingBoxPoints(img.shape[0], img.shape[1], yMin, xMin, yMax, xMax, density=density)
  img[list(yx_set)] = color
  return img


# box_dict: (yMin, xMin, yMax, xMax) -> [(color 3-tuple, density 2-tuple), (...), ...]
def UtilDrawBoxesProbab(img, box_dict):
  # Dictionary of point -> list of its colors
  point_color = collections.defaultdict(lambda: [])

  for (yMin, xMin, yMax, xMax), color_density_list in box_dict.items():
    for color, density in color_density_list:
      yx_set = UtilGetBoundingBoxPoints(img.shape[0], img.shape[1], yMin, xMin, yMax, xMax, density=density)
      for point in yx_set:
        point_color[point].append(color)

  for point, color_list in point_color.items():
    ind = random.randint(0, len(color_list)-1)
    img[point[0], point[1]] = color_list[ind]

  return img


# Returns coordinates of a bounding rectangle, with given density fraction.
# If density = (m, n), then the first m points of each n are painted, the rest are not.
# Painting starts at a position that id divisible by n, so we get regular patterns from
# different overlapping rectangles.
def UtilGetBoundingBoxPoints(imgHeight, imgWidth, yMin, xMin, yMax, xMax, density=(1, 1)):
  point_set = set()

  yMin, xMin, yMax, xMax = [int(round(x)) for x in (yMin, xMin, yMax, xMax)]

  def _brokenLine(start, end, density=density):
    s = set()
    offset = start % density[1]
    for i in range(0, density[0]):
      for j in range(start - offset, end, density[1]):
        val = i + j
        if (val >= start) and (val <= end):
          s.add(val)
    return s

  def _extend_y(yx_set, y, x_set):
    yx_set |= set((y, x) for x in x_set)

  def _extend_x(yx_set, y_set, x):
    yx_set |= set((y, x) for y in y_set)

  def _getYRange(yx_set, yStart, yEnd, x):
    _extend_x(yx_set, _brokenLine(yStart, yEnd), x)

  def _getXRange(yx_set, y, xStart, xEnd):
    _extend_y(yx_set, y, _brokenLine(xStart, xEnd))

  yMin, yMax = (max(min(y, imgHeight-1), 0) for y in (yMin, yMax))
  xMin, xMax = (max(min(x, imgWidth-1), 0) for x in (xMin, xMax))

  yx_set = set()
  # Rectangle might wrap around
  if xMax < xMin:
    _getXRange(yx_set, yMin, xMin, imgWidth)
    _getXRange(yx_set, yMin, 0, xMax)
    _getXRange(yx_set, yMax, xMin, imgWidth)
    _getXRange(yx_set, yMax, 0, xMax)
  else:
    _getXRange(yx_set, yMin, xMin, xMax)
    _getXRange(yx_set, yMax, xMin, xMax)
  _getYRange(yx_set, yMin, yMax, xMin)
  _getYRange(yx_set, yMin, yMax, xMax)

  return yx_set



def UtilFindDrawnBoundingBox(img, color=(255, 0, 0)):
  """
  Finds a bounding box stored in a non-lossy format (like PNG).
  This function, of course, is not guaranteed to find the right rectangle.
  """
  annots = np.all(np.rint(img) == color, axis=-1)
  rowCounts = np.sum(annots, axis=1)
  colCounts = np.sum(annots, axis=0)
  rowArgs = np.argsort(-rowCounts)
  colArgs = np.argsort(-colCounts)
  yMin = np.min(rowArgs[:2])
  yMax = np.max(rowArgs[:2])
  xMin = np.min(colArgs[:2])
  xMax = np.max(colArgs[:2])
  if not (rowCounts[yMin] and rowCounts[yMax] and \
          colCounts[xMin] and colCounts[xMax]):
    # No rectangle of this color
    return None
  highConfidence = True
  if (rowCounts[yMin] != rowCounts[yMax]) or (colCounts[xMin] != colCounts[xMax]):
    highConfidence = False
  return (yMin, xMin, yMax, xMax, highConfidence)


def UtilDisplayMatrixWithCharts(img, imageName=None, chartList=None, colorList=None, asIs=False):
  """
  Elements in chartList must have values within [0, img.shape[0]] range
  :param img:
  :param imageName:
  :param chartList:
  :param colorList:
  :return:
  """
  assert (len(chartList) <= 3) or (len(chartList) == len(colorList))
  assert (colorList is None) or (len(chartList) == len(colorList))
  for chart in chartList:
    assert chart.shape[0] == img.shape[1]
  if colorList is None:
    colorList = [[255, 0, 0], [0, 255, 0], [0, 0, 255]][:len(chartList)]

  if len(img.shape) == 2:
    img = np.stack([img] * 3, axis=2)
  if not asIs:
    img = UtilDbgMatrixToImage(img, imageName=None, method='direct')

  def _adjustChart(c):
    c = np.rint(c).astype(np.int).clip(min=0, max=img.shape[0] - 1)
    c = c[:img.shape[1]]
    return c

  chartList = [_adjustChart(chart) for chart in chartList]
  for chart in chartList:
    img[chart, range(img.shape[1])] = [0, 0, 0]
  for chart in chartList:
    img[chart, range(img.shape[1])] = [0, 0, 0]

  img = img.astype(np.int)
  for ind, chart in enumerate(chartList):
    img[chart, range(img.shape[1])] += np.array(colorList[ind], dtype=np.int)
  img = img.clip(min=0, max=255).astype(np.uint8)

  if imageName is not None:
    UtilArrayToImageFile(img, imageName)
  return img






