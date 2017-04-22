# Enable forward compatibility with Python 3.x

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
usingVersion3 = (sys.version_info[0] == 3)

if not usingVersion3:
    from future.utils import viewitems
    reduce = reduce
    maxint = sys.maxint
else:
    from functools import reduce
    maxint = sys.maxsize

