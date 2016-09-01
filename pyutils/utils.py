# Shared utilities and classes

import os
import json
try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict as DefDict
from PIL import Image

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
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

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


def UtilJSONEncoderFactory(progrIndPeriod = 10000):

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

def UtilJsonDecoderFactory(progrIndPeriod = 10000):
    progrIndCount = [0]

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


def UtilDrawHistogram(inputList = None, show = True, bwFactor = None):
    if not inputList:
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
        yCoord = gkde.evaluate(xCoord)
        plt.plot(xCoord, yCoord)
        if show:
            plt.show()

def UtilStore(obj, fileName, progrIndPeriod = 10000):
    _, ext = os.path.splitext(fileName)
    if ext.lower() == ".json":
        json.dump(obj, open(fileName, 'wt'),
            cls = UtilJSONEncoderFactory(progrIndPeriod),
            sort_keys=True, indent=4, ensure_ascii = False)
        print("") # Next line
        return
    if ext.lower() == ".pck":
        pickle.dump(obj, open(fileName, 'wb'))
        return
    raise ValueError("Bad file name %s" % fileName)

def UtilLoad(fileName, progrIndPeriod = 10000):
    _, ext = os.path.splitext(fileName)
    if ext.lower() == ".json":
        obj = json.load(open(fileName, 'rt'),
            object_hook = UtilJsonDecoderFactory(progrIndPeriod))
        print("") # Next line
        return obj
    if ext.lower() == ".pck":
        obj = pickle.load(open(fileName, 'rb'))
        return obj
    raise ValueError("Bad file name %s" % fileName)

def UtilStitchImagesHor(imgNameList, outImageName):
    imgList = []
    for name in imgNameList:
        imgList.append(Image.open(name))

    width = 0
    height = 0
    for image in imgList:
        w, h = image.size
        width += w
        if height < h:
            height = h

    start = 0
    result = Image.new('RGB', (width, height))
    for image in imgList:
        result.paste(im=image, box=(start, 0))
        start += image.size[0]

    result.save(outImageName)

# Wrapper for primitive values, so they can be returned as
# writable pointers from a list
class UtilWrapper:
    def __init__(self, val):
        self.value = val
