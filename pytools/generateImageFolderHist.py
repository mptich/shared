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

# Utility that generates a dictionary of file name -> histogram,
# and saves it in a file. All files with a given extension under
# a givedn directory are checked.

import sys
import os
import numpy as np
import dill
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
	help = "Root directory of images")
ap.add_argument("-o", "--output", required = True,
        help = "Output PKL file name")
ap.add_argument("-e", "--ext", required = True,
        nargs = "+", help = "List of extensions to include")
args = ap.parse_args()

outDict = {}

normInputLen = len(os.path.normpath(args.input))
for root, dirs, files in os.walk(args.input):
  normRoot = os.path.normpath(root)
  rootKey = normRoot[normInputLen:] + '/'

  for fn in files:
    _, ext = os.path.splitext(fn)
    if ext[1:] not in args.ext:
      continue
    key = rootKey + fn
    img = cv2.imread(root+'/'+fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16],
      [0, 256, 0, 256, 0, 256])
    outDict[key] = hist
    count = len(outDict)
    if count % 1000 == 0:
      print("Processed %d images" % count)

print("Processed %d images" % len(outDict))
with open(args.output, 'wb') as fout:
  dill.dump(outDict, fout)

