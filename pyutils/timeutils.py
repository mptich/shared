# Time and interval utilities
#
# Copyright (C) 2008-2017  Author: Misha Orel
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>


from dateutil import parser as DateParser
from datetime import datetime
import time
from shared.pyutils.utils import *


@UtilStaticVars(epochStart=datetime.utcfromtimestamp(0))
def UtilAsciiTstampToSec(asciiTstamp):
    """
    Translates ASCII timestamp to fractional seconds since 1970
    :param asciiTstamp:
    :return:
    """
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


