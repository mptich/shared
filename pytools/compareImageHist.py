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
    
