# Shared utilities and classes
#
# Copyright (C) 2008-2018  Author: Misha Orel
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

# Utility to resize all images ina  directory

import sys
import csv
import glob

from shared.pyutils.imageutils import *


def _usage():
  print('resize_image_files.py <input_dir> <output_dir> <HxW>')
  sys.exit() 

try:
  inputDir = sys.argv[1]
  outputDir = sys.argv[2]
  reso = sys.argv[3]
  height, width = (int(x) for x in reso.split('x'))
except:
  _usage()

counter = 0
for fn in glob.glob(inputDir + '/*.jpg'):
  _, fnBase = os.path.split(fn)
  outFn = outputDir + '/' + fnBase 
  img = UtilImageFileToArray(fn)
  if img is None:
    print('CORRUPTED: %s' % fn)
    continue
  img = UtilImageResize(img, height, width)
  UtilArrayToImageFile(img, outFn)
  counter += 1
  if counter % 1000 == 0:
    print(counter)
