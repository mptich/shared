# Utilities for fanning multiprocess jobs

import shared.pyutils.forwardCompat as forwardCompat
import sys
from shared.pyutils.utils import *
import tempfile
import csv
from multiprocessing import cpu_count, Process
import importlib

def _fanMultiProcessCall(moduleName, funcName, logFileName, *args):
    if logFileName is not None:
        sys.stdout = sys.stderr = open(logFileName, 'w')
    func = getattr(importlib.import_module(moduleName), funcName)
    func(*args)


def UtilFanMultiProcess(moduleName, funcName, listOfArgLists, logFilePrefix=None):
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
        pList.append(Process(target=_fanMultiProcessCall, args=(moduleName, funcName, logFileName) + tuple(argList)))

    # Wait for all child processes to finish
    [p.wait() for p in pList]
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
                           logFilePrefix=None):
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

        exitCodes = UtilFanMultiProcess(moduleName, funcName, listOfArgLists, logFilePrefix=logFilePrefix)
        print ("Child processes exit codes %s" % str(exitCodes))

    finally:
        # Remove temporary files
        fileList = parCsv.finish()
        for fn in fileList:
            os.remove(fn)

    return exitCodes

