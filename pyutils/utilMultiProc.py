# Utilities for fanning multiprocess jobs
#
# Copyright (C) 2016-2018  Author: Misha Orel
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


import shared.pyutils.forwardCompat as forwardCompat
import sys
from shared.pyutils.utils import *
from shared.pyutils.tensorutils import *
import tempfile
import csv
import random
from multiprocessing import cpu_count, Process, Array
import multiprocessing
import shlex
import subprocess
import time
from queue import Queue
from threading import Thread, Lock

def _fanMultiProcessCall(moduleName, func, logFileName, seed, *args):
    # Initialize random generators for each thread separately
    # func is actual local function, or function name if it is in another module`
    random.seed(seed)
    np.random.seed(seed)
    if logFileName is not None:
        sys.stdout = sys.stderr = open(logFileName, 'w', 1)
    if moduleName is not None:
        assert isinstance(func, str)
        module = UtilLoadModuleByPath(moduleName)
        func = getattr(module, func)
    func(*args)


def UtilFanMultiProcess(moduleName, funcName, listOfArgLists, logFilePrefix=None, waitForAll=True):
    """
    Launches prcesses by calling func in different address space
    :param moduleName: name of the module where this function resides
    :param func: name of teh function function to call
    :param listOfArgLists: list of lists containing arguments to call for the function
    :param logFilePrefix: prefix of teh log file; an ordinal is added to it, then ".txt"
    :return: list of exit codes
    """
    pList = []
    for index,argList in enumerate(listOfArgLists):
        logFileName = None
        if logFilePrefix is not None:
            logFileName = logFilePrefix + ("%05d" % index) + ".txt"
        pList.append(Process(target=_fanMultiProcessCall, args=(moduleName, funcName, logFileName, index) + tuple(argList)))

    # Wait for all child processes to finish
    [p.start() for p in pList]

    if waitForAll:
        [p.join() for p in pList]
    else:
        pFinishedList = []
        pRunningList = []
        terminating = False
        while pList:
            for p in pList:
                if p.exitcode is not None:
                    pFinishedList.append(p)
                    if (p.exitcode != 0) and (not terminating):
                        for pr in pList:
                            if pr != p:
                                pr.terminate()
                        terminating = True
                else:
                    pRunningList.append(p)
            pList = pRunningList
            pRunningList = []
            time.sleep(1.)
        pList = pFinishedList

    return [p.exitcode for p in pList]


class UtilMultiCsvWriter(UtilObject):
    """
    This class distributes multiple CSV writes among several files, so they can be eventually
    passed to different processes
    """
    def __init__(self, fileCount, dir=None):
        self.fhs = []
        self.fds = []
        self.fileNames = []
        self.cws = []
        if dir is None:
            dir = tempfile.gettempdir()
        for i in range(fileCount):
            fd, fileName = tempfile.mkstemp(prefix="UtilMultiCsvWriter_", dir=dir, text=True)
            self.fds.append(fd)
            self.fileNames.append(fileName)
            fh = open(fileName, 'w')
            csvw = csv.writer(fh)
            self.cws.append(csvw)
            self.fhs.append(fh)
        self.count = fileCount
        self.counter = 0
        self.active = True

    def getFileNames(self):
        return self.fileNames

    def record(self, l):
        self.cws[self.counter].writerow(l)
        self.counter = (self.counter + 1) % self.count

    def finish(self):
        """
        :return: list of created file names
        """
        if self.active:
            for fh in self.fhs:
                fh.close()
            for fd in self.fds:
                os.close(fd)
            self.active = False
        return self.fileNames


def UtilFanMultiCsvProcess(generator, moduleName, funcName, args=None, parallelCount=None, debugDirectCall=None, \
                           logFilePrefix=None, waitForAll=True):
    """
    Calls generator to generate input CSV for each process. The function takes CSV file name as a first parameter,
    then takes a list of args
    :param generator: returns items for the input CSVs
    :param moduleName: name of the module to load
    :param funcName: name of the function in the module to call
    :param args: list of the addtitional arguments (after input CSV name), all should be strings
    :param debugDirectCall: for debugging, call the function directly, in this process
    :param parallelCount: number of parallel processes. If None, twice the number of CPUs is used
    :param logFilePrefix: first part (with full path) of the log files. If None - no logs are created
    :return: list of exit codes from the processes
    """
    if parallelCount is None:
        parallelCount = cpu_count()

    if args is None:
        args = []

    try:
        parCsv = UtilMultiCsvWriter(parallelCount)
        fileList = parCsv.getFileNames()
        for csvLine in generator:
            parCsv.record(csvLine)
        parCsv.finish()
        print("Created files with file lists: %s" % str(fileList))

        # For debugging
        if debugDirectCall is not None:
            for fn in fileList:
                debugDirectCall(fn, *args)
            sys.exit(0)

        # Start subprocesses
        listOfArgLists = []
        for fn in fileList:
            l = [fn] + args
            listOfArgLists.append(l)

        exitCodes = UtilFanMultiProcess(moduleName, funcName, listOfArgLists, logFilePrefix=logFilePrefix,
                                        waitForAll=waitForAll)
        print ("Child processes exit codes %s" % str(exitCodes))

    finally:
        # Remove temporary files
        fileList = parCsv.finish()
        for fn in fileList:
            os.remove(fn)

    return np.array(exitCodes)


class UtilParallelFixedWriter(UtilObject):
    """
    Writing to a file from a parallel process
    """

    def __init__(self, typeShapeList, entryCount=1):
        """
        :param typeShapeList: list of tuples describing one entry to write: (Numpy type, numpy shape)
        :param entryCount: how many times typeShapeList entries will be repeated
        """
        self.typeShapeList = typeShapeList
        self.size = UtilNumpyEntriesSize(typeShapeList) * entryCount
        self.buffers = []
        for _ in range(2):
            self.buffers.append(Array('B', self.size, lock=False))
        self.currentBuffer = 0
        self.bufPos = 0
        self.process = None

    def addRecord(self, arr):
        """
        Adds a record, as a numpy array
        :param arr: numpy array
        :return: True if buffer is full, False otherwise
        """
        entrySize = UtilNumpyEntryItemSize((arr.dtype, arr.shape))
        assert self.bufPos + entrySize <= self.size
        self.buffers[self.currentBuffer][self.bufPos:self.bufPos + entrySize] = arr.tostring()
        self.bufPos += entrySize
        return self.bufPos == self.size

    def addRecordList(self, arrList):
        """
        Same as addRecord(), but adds all records from the list
        :param arrList:
        :return:
        """
        for arr in arrList:
            self.addRecord(arr)
        return self.bufPos == self.size

    def waitForProc(self):
        if self.process is not None:
            self.process.join()
            if self.process.exitcode != 0:
                raise IOError('Process writing to %s ecxitcode %d' % (self.fileName, self.process.exitcode))

    def write(self, fileName, append=False, postProcessCommand=None):
        self.waitForProc()
        self.fileName = fileName
        self.process = Process(target=self.worker,
                               kwargs={'fileName': fileName, 'array': self.buffers[self.currentBuffer], \
                                       'append': append, 'postProcessCommand': postProcessCommand})
        self.process.start()
        self.currentBuffer ^= 1
        self.bufPos = 0

    def close(self):
        self.waitForProc()

    @staticmethod
    def worker(fileName, array, append, postProcessCommand):
        path, _ = os.path.split(fileName)
        UtilSafeMkdir(path)
        mode = 'ab' if append else 'wb'
        with open(fileName, mode) as f:
            f.write(array)
        if postProcessCommand is not None:
            args = shlex.split(postProcessCommand)
            subprocess.Popen(args)


class UtilMultithreadQueue:
    """
    Multithreading fetching of objects
    """
    def __init__(self, state, func, threadCount=cpu_count(), maxQueueSize=1000):
        self.on_ = True
        self.state_ = state
        self.func_ = func
        self.que_ = Queue(maxsize=maxQueueSize)
        self.lock_ = Lock()
        self.threads_ = []
        for i in range(threadCount):
            t = Thread(target=self.worker, args=(i,))
            self.threads_.append(t)
            t.start()

    def worker(self, seed):
        # Initialize random generators for each thread separately
        random.seed(seed)
        np.random.seed(seed)
        while self.on_:
            ret = self.func_(self.state_, self.lock_)
            if ret is None:
                return
            self.que_.put(ret)

    def getData(self):
        return self.que_.get()

    def checkData(self):
        return not self.que_.empty()

    def terminate(self):
        self.on_ = False
        while self.checkData():
            self.getData()


class UtilMultiprocQueue:
    """
    Multiproc fetching of objects
    """
    def __init__(self, func, name="", procCount=cpu_count(),
        logFileDir=None, maxQueueSize=1000):
        self.on_ = True
        self.state_ = None
        self.func_ = func
        self.name_ = name
        self.manager_ = multiprocessing.Manager()
        self.que_ = self.manager_.Queue(maxsize=maxQueueSize)
        self.logFileDir_ = logFileDir;
        self.lock_ = self.manager_.Lock()
        self.procs_ = []
        for i in range(procCount):
            p = Process(target=self.worker, args=(i,))
            self.procs_.append(p)

    def start(self):
      for p in self.procs_:
        p.start()

    def setState(self, state):
      self.state_ = state

    def worker(self, index):
        # Initialize random generators for each thread separately
        random.seed(index)
        np.random.seed(index)
        if self.logFileDir_ is not None:
          sys.stdout = sys.stderr = open(
            self.logFileDir_ + ('/%04d_%s.txt' % (index, self.name_)), 'w', 1)
        while self.on_:
            ret, val = self.func_(self.state_, self.lock_)
            if ret is None:
                return
            if not ret:
              # Yield control and continue, no data for the queue
              time.sleep(0.)
              continue
            self.que_.put(val)

    def getData(self):
        return self.que_.get()

    def checkData(self):
        return not self.que_.empty()

    def lock(self):
      self.lock_.acquire()

    def unlock(self):
      self.lock_.release()

    def getManager(self):
      return self.manager_

    def terminate(self):
        self.on_ = False
        while self.checkData():
            self.getData()
        for p in self.procs_:
          p.join()



class UtilQuickParallelProc:
  def __init__(self, maxProcCount = 2 * cpu_count(), terminateOnError=True):
    self.maxProcCount_ = maxProcCount
    self.procList_ = []
    self.cbDict_ = {}
    self.logDict_ = {}
    self.terminateOnError_ = terminateOnError

  def run(self, func, argList, logFileName=None, callback=None, cbArg=None):
    # Might block
    self.waitForCount(self.maxProcCount_ - 1)

    seed = np.random.randint(low=0, high=1000000)
    p = Process(target=_fanMultiProcessCall, args=(None, func, logFileName, seed) + tuple(argList))
    p.start()
    self.logDict_[p.pid] = logFileName
    if callback is not None:
      self.cbDict_[p.pid] = (callback, cbArg)
    self.procList_.append(p)

  def join(self):
    # Wait for termination
    self.waitForCount(0)

  def waitForCount(self, count):
    while len(self.procList_) > count:
      # Yield control
      time.sleep(0)
      
      removeList = []
      for p in self.procList_:
        if not p.is_alive():
          if p.exitcode != 0:
            errStr = "Process id %d log file %s failed" % \
              (p.pid, self.logDict_[p.pid])
            if self.terminateOnError_:
              raise Exception(errStr)
            else:
              print("ERROR: %s" % errStr)
              self.terminate()
              return False
          removeList.append(p)

      for p in removeList:
        if p.pid in self.cbDict_:
          callback, cbArg = self.cbDict_[p.pid]
          del self.cbDict_[p.pid]
          callback(cbArg)
        del self.logDict_[p.pid]
        self.procList_.remove(p)

    return True


  def terminate(self):
    # Forcefully terminate all processes
    for p in self.procList_:
      if p.is_alive():
        p.terminate()

    while True:
      finished = True
      for p in self.procList_:
        if p.is_alive():
          finished = False
      if finished:
        break
      # Yield control
      time.sleep(0)









