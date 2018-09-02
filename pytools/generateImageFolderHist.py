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

