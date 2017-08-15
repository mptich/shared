# Enable forward compatibility with Python 3.x

# Copyright (C) 2015-2017  Author: Misha Orel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import platform
usingVersion3 = (sys.version_info[0] == 3)

if not usingVersion3:
    from future.utils import viewitems
    reduce = reduce
    maxint = sys.maxint
else:
    from functools import reduce
    maxint = sys.maxsize

import numpy as np
np.seterr(all='raise')

def ThisIsWindows():
    return platform.system().lower() == 'windows'
