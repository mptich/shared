# Shared utilities and classes

import json

UtilObjectKey = "__utilobjectkey__"
UtilSetKey = "__utilsetkey__"

# Simple hashable dictionary
class UtilDict(dict):
    def __hash__(self):
        return hash(tuple([(k, tuple(v)) for (k,v) in sorted(self.items())]))

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
            for k, v in d.items():
                setattr(self, k, v)
            return True
        return False


    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.key() == other.key()

    def __ne__(self, other):
        return self.key() != other.key()

    def __le__(self, other):
        return not self.key() > other.key()

    def __gt__(self, other):
        return self.key() > other.key()

    def __hash__(self):
        return hash(self.key())


class UtilJSONEncoder(json.JSONEncoder):
    """
    Converts a python object, where the object is derived from UtilObject,
    into an object that can be decoded using the GenomeJSONDecoder.
    """
    def default(self, obj):
        if isinstance(obj, UtilObject):
            d = obj.__dict__
            d[UtilObjectKey] = obj.__module__ + '.' + obj.__class__.__name__
            return d
        if isinstance(obj, set):
            d = {}
            d[UtilSetKey] = list(obj)
            return d
        return obj

def UtilJSONDecoderDictToObj(d):
    if UtilObjectKey in d:
        moduleName, _, className = d[UtilObjectKey].rpartition('.')
        assert(moduleName)
        module = __import__(moduleName)
        classType = getattr(module, className)
        kwargs = dict((x.encode('ascii'), y) for x, y in d.items())
        inst = classType(**kwargs)
    elif UtilSetKey in d:
        inst = set(d[UtilSetKey])
    else:
        inst = d
    return inst

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
        self.hitCount = 0
        self.xactCount = 0

    def write(self, fileName, line):
        assert(self.mode[0] in ('w', 'a'))
        f = self.fileHandle(fileName)
        try:
            f.write(line)
        except:
            print("Could not write to %s" % fileName)
            return
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
            except:
                print("Could not open %s" % fileName)
                return None
            self.fileDict[fileName] = f
            self.fileList.append(fileName)
        else:
            self.hitCount += 1
        return self.fileDict[fileName]

    def closeAll(self):
        for f in self.fileDict.values():
            f.close()
        self.fileDict = {}
        self.fileList = []

    def getStats(self):
        return "%u hits out of %u transactions: %u%%" % (self.hitCount,
                self.xactCount, (100 * self.hitCount / self.xactCount) if
                self.xactCount else 0)
