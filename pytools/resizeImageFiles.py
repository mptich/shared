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
