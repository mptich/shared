# Enable forward compatibility with Python 3.x
#
# Copyright (C) 2015-2018  Author: Misha Orel
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


#from __future__ import print_function
#from __future__ import absolute_import
#from __future__ import division

import sys
import platform
import re

if sys.version_info[0] < 3:
    print('MUST BE PYTHON 3')
    sys.exit()

from functools import reduce
maxint = sys.maxsize

import numpy as np
np.seterr(all='raise')

def ThisIsWindows():
    return platform.system().lower() == 'windows'

def VersionCompare(v1, v2):
    def _normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]
    def _compare(x1, x2):
        if x1 > x2:
            return 1
        elif x1 < x2:
            return -1
        else:
            return 0
    return _compare(_normalize(v1), _normalize(v2))
