# Utility that compares 2 dictionaries of image histograms.
# Takes 2 PKL files as input.

import sys
import os
import numpy as np
import dill
import cv2

first = sys.argv[1]
second = sys.argv[2]

with open(first, "rb") as fin:
  d1 = dill.load(fin)
with open(second, "rb") as fin:
  d2 = dill.load(fin)

d = {}
for fn1, h1 in d1.items():
  for fn2, h2 in d2.items():
    tup = (fn1, fn2)
    diff = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    d[tup] = diff
    count = len(d)
    if count % 10000 == 0:
      print("Processed %d" % count)

l = []
for tup, diff in d.items():
  l.append((diff, tup))

l = sorted(l, reverse=True)
for i in range(10000):
  print(l[i])
    
