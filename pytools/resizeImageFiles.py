# Utility to resize all images ina  directory

import sys
import csv
import glob

from shared.pyutils.imageutils import *


def _usage():
  print('resize_image_files.py <input_pattern> <output_dir> <HxW>')
  sys.exit() 

try:
  inputPattern = sys.argv[1]
  outputDir = sys.argv[2]
  reso = sys.argv[3]
  height, width = (int(x) for x in reso.split('x'))
except:
  _usage()

for fn in glob.glob(input_pattern):
  _, fnBase = os.path.split(fn)
  outFn = outputDir + '/' + fnBase 
  img = UtilImageFileToArray(fn)
  img = UtilImageResize(img, height, width)
  UtilArrayToImageFile(img, outFn)
