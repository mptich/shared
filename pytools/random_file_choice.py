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

# If "ext" is specified, files are moved so that extensions are kept consistent.
 
import glob
import shutil
import random
import sys
import os

def _usage():
    print('USAGE: random_file_choice.py <source dir> <dest dir> <count> [ext] [move]')
    sys.exit(0)

try:
    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    count = int(sys.argv[3])
    optionals = sys.argv[4:]

    move = False
    useExt = False
    for o in optionals:
      if o == 'move':
        move = True
      if o == 'ext':
        useExt = True
except:
    _usage()
    
file_set = set(glob.glob(src_dir + '/*'))
if useExt:
  base_set = set()
  ext_set = set()
  for fn in file_set:
    base, ext = os.path.splitext(fn)
    base_set.add(base)
    ext_set.add(ext)
  file_set = base_set
else:
  ext_set = set([""])

file_list = list(file_set)
random.shuffle(file_list)
file_list = file_list[:count]

for fn in file_list:
    _, fn_base = os.path.split(fn)
    fn_out = dest_dir + '/' + fn_base
    for ext in ext_set:
      fn_out_full = fn_out + ext
      fn_full = fn + ext
      if move:
        shutil.move(fn_full, fn_out_full)
      else:
        shutil.copy(fn_full, fn_out_full)
