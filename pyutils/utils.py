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


import os
import sys
import json
import glob
import traceback
import types
import shared.pyutils.forwardCompat as forwardCompat
try:
   import cPickle as pickle
except:
   import pickle
import matplotlib.pyplot as plt
# To make it compatible with ASCII only environment
if 'PYPLOT_WITHOUT_DISPLAY' in os.environ:
    plt.switch_backend('agg')
from collections import defaultdict as DefDict
import errno
from subprocess import Popen, PIPE


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
        obj = json.load(open(fileName, 'r', encoding='utf-8'),
            object_hook = UtilJsonDecoderFactory(progrIndPeriod=progrIndPeriod, retList=ctrList))
        if ctrList[0][0] > 0:
            print("") # Next line
        return obj

    if fileType == "PICKLE":
        obj = pickle.load(open(fileName, 'rb'))
        return obj

    raise ValueError("Bad file name %s" % fileName)


def UtilStaticVars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


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


class UtilTemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp() so it's usable with "with" statement."""
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)
        
        
class UtilNotebookLoader(object):
    """
    Module Loader for Jupyter Notebooks
    The idea is taken from:
    http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Importing%20Notebooks.ipynb
    """
    
    # This class should only work in ipython environment. So if we cannot import ipython, we should silently fail
    try:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython import get_ipython
        from nbformat import read
        valid = True
    except ImportError as ex:
        valid = False

    
    def __init__(self, path, name=None):
        
        if not UtilNotebookLoader.valid:
            # Wrong environment
            self.mod = None
            return
        
        moduleName = path if name is None else name
        
        # Check if module is already loaded
        if moduleName in sys.modules:
            self.mod = sys.modules[moduleName]
            return
        
        self.shell = UtilNotebookLoader.InteractiveShell.instance()
        self.path = path
           
        print ("importing Jupyter notebook from %s" % path)
                                       
        # load the notebook object
        with open(path, 'r', encoding='utf-8') as f:
            nb = UtilNotebookLoader.read(f, 4)
        
        # create the module and add it to sys.modules
        mod = types.ModuleType(moduleName)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__['get_ipython'] = UtilNotebookLoader.get_ipython
        sys.modules[moduleName] = mod
        
        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__
        
        try:
          for cell in nb.cells:
            if cell.cell_type == 'code':
                # transform the input to executable Python
                code = self.shell.input_transformer_manager.transform_cell(cell.source)
                # run the code in themodule
                exec(code, mod.__dict__)
        finally:
            self.shell.user_ns = save_user_ns
            
        self.mod = mod
        
    def module(self):
        return self.mod
        

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


def UtilGetModuleDir(name):
    """
    For compatobility with Windows, where sys might not reurn an absolute path
    :param name:
    :return:
    """
    return os.path.normpath(os.path.split(sys.modules[name].__file__)[0])


class UtilSimpleLinkedList(UtilObject):
    """
    Works for any object with prev / next attributes reserved for this list (simpler and faster than using Node,
    but unusable if an object is in > 1 linked list)
    """

    def __init__(self):
        self.head = None
        self.tail = None
        self._count = 0

    def append(self, obj):
        self._count += 1
        temp, self.tail = (self.tail, obj)
        obj.next = None
        obj.prev = temp
        if temp is not None:
            temp.next = obj
        else:
            self.head = obj

    def dequeue(self, obj):
        if obj.next is not None:
            obj.next.prev = obj.prev
        else:
            self.tail = obj.prev
        if obj.prev is not None:
            obj.prev.next = obj.next
        else:
            self.head = obj.next
        self._count -= 1

    def count(self):
        return self._count


def UtilShellCmd(cmd, printCmd=True, printStdout=True, printStderr=True):
    if printCmd:
        print(cmd)
    try:
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
    except Exception as ex:
        out = b""
        err = bytes(str(ex), 'utf-8')
    if printStdout and out:
        print(out.decode('utf-8', 'ignore'), end='', file=sys.stdout)
    if printStderr and err:
        print(err.decode('utf-8', 'ignore'), end='', file=sys.stderr)
    return (out, err)


# Wrapper for primitive values
class UtilWrapper:
    def __init__(self, val):
        self.value = val


class TracePrints(object):
    """
    Tracing stray prints
    adapted and enhanced from https://stackoverflow.com/questions/1617494/finding-a-print-statement-in-python
    """
    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(f=sys._getframe(1), limit=1, file=self.stdout)

    def flush(self):
        self.stdout.flush()






