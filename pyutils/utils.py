# Shared utilities and classes
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import json
import glob
try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict as DefDict
import errno
import functools
from dateutil import parser as DateParser
from datetime import datetime


UtilObjectKey = "__utilobjectkey__"
UtilSetKey = "__utilsetkey__"
UtilJsonDumpMethod = "utilJsonDump"
UtilJsonLoadMethod = "utilJsonLoad"
UtilDumpKey = "__utildumpkey__"

# Simple hashable dictionary
class UtilDict(dict):
    def __hash__(self):
        return hash(tuple([(k, tuple(v)) for (k,v) in
            sorted(self.iteritems())]))

class UtilError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class UtilObject(object):
    """
    Base class defining serialization methods.
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def buildFromDict(self, d):
        if UtilObjectKey in d:
            d.pop(UtilObjectKey)
            for k, v in d.iteritems():
                setattr(self, k, v)
            return True
        return False


    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else repr(type(o)),
                sort_keys=True)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key

    def __le__(self, other):
        return not self.key > other.key

    def __gt__(self, other):
        return self.key > other.key

    def __hash__(self):
        return hash(self.key)

# Universal function creator
class UtilCaller(UtilObject):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        d = self.kwargs.copy()
        d.update(kwargs)
        return self.func(*(self.args + args), **d)

def UtilIdentity(*args):
    if len(args) == 1:
        return args[0]
    return args

def UtilJSONEncoderFactory(progrIndPeriod = 10000, retList = None):
    progrIndCounter = [0]
    if retList is not None:
        retList.append(progrIndCounter)

    class UtilJSONEncoder(json.JSONEncoder):
        """
        Converts a python object, where the object is derived
        from UtilObject, into an object that can be decoded
        using the GenomeJSONDecoder.
        """

        def __init__(self, *args, **kwargs):
            super(UtilJSONEncoder, self).__init__(*args, **kwargs)
            self.progrIndCount = 0

        def default(self, obj):
            self.progrIndCount += 1
            if self.progrIndCount % progrIndPeriod == 0:
                print ("\rProgress %u" % self.progrIndCount),
                progrIndCounter[0] += 1

            # First see if there is an alternative method to represent the
            # class
            if getattr(obj, UtilJsonDumpMethod, None):
                d = {}
                d[UtilObjectKey] =\
                    obj.__module__ + '.' + obj.__class__.__name__
                d[UtilDumpKey] = getattr(obj, UtilJsonDumpMethod)()
                return d

            # Generic method for UtilObject instamces
            if isinstance(obj, UtilObject):
                d = obj.__dict__
                d[UtilObjectKey] =\
                    obj.__module__ + '.' + obj.__class__.__name__
                return d

            # Method for sets (not represented in JSON)
            if isinstance(obj, set):
                d = {}
                d[UtilSetKey] = list(obj)
                return d
            return obj

    return UtilJSONEncoder


# This is optimization - caching imported classes
importedClassesDict = {}

def UtilJsonDecoderFactory(progrIndPeriod = 10000, retList = None):
    progrIndCount = [0]
    progrIndCounter = [0]

    if retList is not None:
        retList.append(progrIndCounter)

    def UtilJSONDecoderDictToObj(d):
        if UtilObjectKey in d:
            fullClassName = d[UtilObjectKey]
            if fullClassName in importedClassesDict:
                classType = importedClassesDict[fullClassName]
            else:
                moduleName, _, className = fullClassName.rpartition('.')
                assert(moduleName)
                module = __import__(moduleName)
                classType = getattr(module, className)
                importedClassesDict[fullClassName] = classType
            kwargs = dict((x.encode('ascii'), y) for x, y in d.iteritems())
            inst = classType(**kwargs)
        elif UtilSetKey in d:
            inst = set(d[UtilSetKey])
        else:
            inst = d
        progrIndCount[0] += 1
        if progrIndCount[0] % progrIndPeriod == 0:
            print ("\rProgress %u" % progrIndCount[0]),
            progrIndCounter[0] += 1
        return inst

    return UtilJSONDecoderDictToObj


class UtilMultiFile(UtilObject):
    """
    Keeps specified number of files opened, for read or write
    Attributes:
        mode - read or write
        maxCount - maximum number of opened files at any given moment
        hitCount - number of open file hits
        xactCount - number of transactions (reads or writes)
        fileList - list of open file names, sorted by the time
        fileDict - map of file name to a file handle
    """

    def __init__(self, maxCount, mode):
        self.maxCount = maxCount
        self.mode = mode
        self.fileList = []
        self.fileDict = {}
        self.fileCache = DefDict(list)
        self.hitCount = 0
        self.xactCount = 0

    def write(self, fileName, line):
        # Try to cache it first
        lines = self.fileCache[fileName]
        lines.append(line)
        if len(lines) > 100:
            self.cacheFlush(fileName)

    def cacheFlush(self, fileName):
        assert(self.mode[0] in ('w', 'a'))
        f = self.fileHandle(fileName)
        for l in self.fileCache[fileName]:
            try:
                f.write(l)
            except IOError as e:
                print("Could not write to %s: error %d %s" % (
                    fileName, e.errno, e.strerror))
                return
        self.fileCache[fileName] = []
        self.xactCount += 1

    def fileHandle(self, fileName):
        if fileName not in self.fileDict:
            if len(self.fileList) == self.maxCount:
                oldFileName = self.fileList[0]
                self.fileDict[oldFileName].close()
                del self.fileDict[oldFileName]
                self.fileList = self.fileList[1:]
            try:
                f = open(fileName, self.mode)
            except IOError as e:
                print("Could not open %s: error %d %s" % (fileName, e.errno,
                                                          e.strerror))
                return None
            self.fileDict[fileName] = f
            self.fileList.append(fileName)
        else:
            self.hitCount += 1
        return self.fileDict[fileName]

    def closeAll(self):
        for fileName in self.fileCache.keys():
            self.cacheFlush(fileName)
        for f in self.fileDict.values():
            f.close()
        self.fileDict = {}
        self.fileList = []

    def getStats(self):
        return "%u hits out of %u transactions: %u%%" % (self.hitCount,
                self.xactCount, (100 * self.hitCount / self.xactCount) if
                self.xactCount else 0)


def UtilDrawHistogram(inputList = None, show = True, bwFactor = None, saveFile = None):
    if inputList is None:
        if show:
            plt.show()
        return
    input = np.array(sorted(inputList))
    gkde = None
    start = input[0]
    stop = input[-1]
    if start != stop:
        step = (stop - start) / 200.
        if bwFactor:
            bandwidth = (stop - start) / bwFactor
        else:
            bandwidth = 'silverman'
    else:
        start = start - 1
        stop = start + 1
        step = 1.0
        bandwidth = 1.0
    try:
        gkde = stats.gaussian_kde(input, bw_method=bandwidth)
    except:
        print ("gaussian_kde failed on list %s" % repr(inputList))
    if gkde:
        xCoord = np.arange(start, stop, step)
        try:
            yCoord = gkde.evaluate(xCoord)
        except FloatingPointError as ex:
            print('UtilDrawHistogram: xCoord=%s: %s' % (str(xCoord), str(ex)))
            return
        plt.plot(xCoord, yCoord)
        if show:
            plt.show()
        if saveFile is not None:
            plt.savefig(saveFile)

def UtilCloseHistogram():
    plt.close()

def UtilMergeDicts(*dictArgs):
    """
    This function "fixes" the fact that dict.update() returns None
    """
    ret = {}
    for d in dictArgs:
        ret.update(d)
    return ret

def UtilStorageFileType(fileName):
    # By extension
    fileType = None
    _, ext = os.path.splitext(fileName)
    if ext.lower() == ".json":
        fileType = "JSON"
    if ext.lower() == ".pck":
        fileType = "PICKLE"
    return fileType

def UtilStore(obj, fileName, progrIndPeriod = 10000, fileType=None):
    if fileType is None:
        fileType = UtilStorageFileType(fileName)

    if fileType == "JSON":
        ctrList = []
        json.dump(obj, open(fileName, 'wt'),
            cls = UtilJSONEncoderFactory(progrIndPeriod, retList = ctrList),
            sort_keys=True, indent=4, ensure_ascii = False)
        if ctrList[0][0] > 0:
            print("") # Next line
        return

    if fileType == "PICKLE":
        pickle.dump(obj, open(fileName, 'wb'))
        return

    raise ValueError("Bad file name %s" % fileName)

def UtilLoad(fileName, progrIndPeriod = 10000, fileType=None):
    if fileType is None:
        fileType = UtilStorageFileType(fileName)

    if fileType == "JSON":
        ctrList = []
        obj = json.load(open(fileName, 'rt'),
            object_hook = UtilJsonDecoderFactory(progrIndPeriod=progrIndPeriod, retList=ctrList))
        if ctrList[0][0] > 0:
            print("") # Next line
        return obj

    if fileType == "PICKLE":
        obj = pickle.load(open(fileName, 'rb'))
        return obj

    raise ValueError("Bad file name %s" % fileName)

def BivarPolynomialOffset(coefList, dx, dy):
    """
    Given an input polinomial of x and y f(x,y), calculate coefficients for f(x+dx,y+dy).
     Taylor function is used for this
    :param coefList: list of polynomyal coeffecients, in the order A22 A12 A02 A11 A01 A00, where
        Amn - mth degree of x, (n-m)th degree of y
    :param dx: offset along x
    :param dy: offset along y
    :return: list of updated coefficients, taking into accound dx and dy
    """

    # N is max polinomial degree. len(coefList) = (N+1)*(N+2)/2
    N = 0
    while ((N+1) * (N+2) / 2) < len(coefList):
        N += 1
    if ((N+1) * (N+2) / 2) > len(coefList):
        raise ValueError("BivarPolynomialOffset wrong coefList length %d" % len(coefList))

    c = coefList
    if N == 0:
        return coefList[:]
    if N == 1:
        return [c[0]+c[1]*dy+c[2]*dx, c[1], c[2]]
    if N == 2:
        return [c[0]+c[1]*dy+c[2]*dx+c[3]*dy**2+c[4]*dx*dy+c[5]*dx**2, \
            c[1]+2*c[3]*dy+c[4]*dx, c[2]+2*c[5]*dx+c[4]*dy, c[3], c[4], c[5]]
    if N == 3:
        return [c[0]+c[1]*dy+c[2]*dx+c[3]*dy**2+c[4]*dx*dy+c[5]*dx**2+c[6]*dy**3+c[7]*dx*dy**2+ \
            c[8]*dx**2*dy+c[9]*dx**3, \
            c[1]+2*c[3]*dy+c[4]*dx+3*c[6]*dy**2+2*c[7]*dx*dy+c[8]*dx**2, \
            c[2]+c[4]*dy+2*c[5]*dx+c[7]*dy**2+2*c[8]*dx*dy+3*c[9]*dx**2, \
            c[3]+3*c[6]*dy+c[7]*dx, c[4]+2*c[7]*dy+2*c[8]*dx, c[5]+c[8]*dy+3*c[9]*dx, c[6], c[7], c[8], c[9]]

    raise ValueError("BivarPolynomialOffset degree %d is not supported yet" % N)


def UtilStaticVars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


# It is expensive first time, so implement it as a function
def UtilNumpyClippingValue(dtype):
    info = np.finfo(dtype=dtype)
    return info.tiny * 10.


def UtilSafeMkdir(dirName):
    """
    If several processes are trying to create a directory simultaneously, it might cause a race condition.
    :param dirName: Directory that should be created if it does not exist
    :return: True if directory has been created, False othewise
    """
    ret = False
    try:
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            ret = True
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return ret

def UtilSafeMultiGlob(listOfPatterns):
    """
    In Windows, same file name could be returned several times from multiple globs, because of upper/lower
    case characters in file names. This generator makes sure that each file name returned just once
    :param listOfPatterns: list of patterns to be passed to glob.glob()
    :return:
    """
    s = set()
    for pat in listOfPatterns:
        s |= set(glob.glob(pat))
    for name in s:
        yield name


# Wrapper for primitive values, so they can be returned as
# writable pointers from a list
class UtilWrapper:
    def __init__(self, val):
        self.value = val


def UtilNumpyEntryItemSize(typeShapeTuple):
    """
    Calculates size of a numpy object
    :param typeShapeTuple: (numpy type, numpy shape)
    :return: size
    """
    return np.dtype(typeShapeTuple[0]).itemsize * functools.reduce(lambda x, y: x*y, typeShapeTuple[1])


def UtilNumpyEntriesSize(typeShapeList):
    """
    Calculates size of several numpy objects
    :param typeShapeList: list of tuples (numpy type, numpy shape)
    :return: size
    """
    return sum([UtilNumpyEntryItemSize(x) for x in typeShapeList])


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






