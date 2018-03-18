# Time and interval utilities
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



from dateutil import parser as DateParser
from datetime import datetime
import time
from shared.pyutils.utils import *
from shared.pyutils.tensorutils import UtilNumpyRle


@UtilStaticVars(epochStart=datetime.utcfromtimestamp(0))
def UtilAsciiTstampToSec(asciiTstamp):
    """
    Translates ASCII timestamp to fractional seconds since 1970
    :param asciiTstamp:
    :return:
    """
    # Sometimes % or _ is used as a separator between date and time
    asciiTstamp = asciiTstamp.replace('%', ' ')
    asciiTstamp = asciiTstamp.replace('_', ' ')
    return (DateParser.parse(asciiTstamp) - UtilAsciiTstampToSec.epochStart).total_seconds()


def UtilAsciiTstampToMsec(asciiTstamp):
    return np.int64(UtilAsciiTstampToSec(asciiTstamp) * 1000)


def UtilSecToAsciiTstamp(seconds):
    fractSec = seconds % 1
    s = time.strftime('%m-%d-%Y %H:%M:%S', time.gmtime(int(seconds)))
    fractStr = '.' + ('%06u' % int(fractSec * 1000000))
    return s + fractStr


def UtilMsecToAsciiTstamp(ms):
    return UtilSecToAsciiTstamp(ms / 1000.)


@UtilStaticVars(epochStart=datetime.utcfromtimestamp(0))
def UtilMsecTstamp():
    tstamp = datetime.utcnow()
    int((tstamp - UtilMsecTstamp.epochStart).total_seconds() * 1000)


def UtilTimedRle(data, timeStamps):
    """
    Produces run length encoding lists with timestamps
    See UtilNumpyRle
    :param data: data to RLE
    :param timeStamps: timestamps in any time units
    :return:
    """
    intervals, values = UtilNumpyRle(data)
    intervals = [(timeStamps[start], timeStamps[stop]) for start, stop in intervals]
    return (intervals, values)


