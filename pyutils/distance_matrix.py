# Distance matrix is built on teh top of numpy arrays

from utils import *
import numpy as np
import operator
import copy
import imp
imp.load_source("kendall", "../algorithms/kendall.py")
from kendall import calculateWeightedKendall

class DistanceMatrixRow(UtilObject):
    """
    This class allows implementation of [][] oprator on DistanceMatrix
    """
    def __init__(self, row, distMatrix):
        self.row = row
        self.distMatrix = distMatrix

    def __getitem__(self, itemName):
        matrix = self.distMatrix
        return matrix.getArray()[self.row] \
            [matrix.getNames().indexOf(itemName)]

    def __setitem__(self, itemName, val):
        matrix = self.distMatrix
        matrix.getArray()[self.row][matrix.getNames().index(itemName)] = val

    def __iter__(self):
        for x in self.distMatrix.getArray()[self.row]:
            yield x


class DistanceMatrix(UtilObject):
    def __init__(self, **kwargs):
        if self.buildFromDict(kwargs):
            return

        doubleDict = kwargs.get("doubleDict", None)
        if doubleDict:
            self.convertFromDoubleDict(doubleDict)
            return

        self.size = kwargs["size"]
        self.array = np.array(shape=(self.size, self.size), dtype=float,
                              order='C')
        self.names = sorted(kwargs.get("names", []))

    def convertFromDoubleDict(self, doubleDict):
        self.size = len(doubleDict)
        self.names = sorted(doubleDict.keys())
        temp = [x[1] for x in sorted(doubleDict.items(), key =
            operator.itemgetter(0))]
        self.array = np.array([[y[1] for y in sorted(x.items(), key =
            operator.itemgetter(0))] for x in temp])

    def clone(self):
        return DistanceMatrix(size = self.size, names = self.names)

    def addRow(self, rowNumber, rowList):
        assert(len(rowList) == self.size)
        assert(rowNumber < self.size)
        self.array[: rowNumber] = rowList

    def addName(selfself, index, name):
        assert(index < self.size)
        self.names[index] = name

    def getSize(self):
        return self.size

    def getNames(self):
        return self.names

    def getName(self, index):
        return self.names[index]

    def getArray(self):
        return self.array

    def __getitem__(self, itemId):
        if isinstance(itemId, (int, long)):
            return self.array[itemId]
        return DistanceMatrixRow(self.names.index(itemId), self)


def distanceMatrixCorrelation(matrix1, matrix2, weights):
    """
    :param matrix1:
    :param matrix2:
    :param weights:
    :return: mean, and STD of the Kendal Tau Distances between all rows,
    and sorted list of names in the order of better correlations
    """

    size = matrix1.getSize()
    assert(size == matrix2.getSize())
    assert((not weights) or (size == weights.getSize))
    kendallList = [None] * size
    weightsAllOnes = [1.0] * size
    for i in range(size):
        kendallList[i] = calculateWeightedKendall(list(matrix1[i]),
            list(matrix2[i]), list(weights[i]) if weights else weightsAllOnes)
    sortedNames = sorted(zip(matrix1.names, kendallList), key =
        operator.itemgetter(1))
    return (np.mean(kendallList), np.std(kendallList), sortedNames)


# Test
if __name__ == "__main__":
    dd={'a':{}, 'b':{}, 'c':{}}
    dd['a']['a'] = 1.0
    dd['a']['b'] = 2.5
    dd['a']['c'] = 4.0
    dd['b']['a'] = 2.5
    dd['b']['b'] = 1.0
    dd['b']['c'] = 2.2
    dd['c']['a'] = 4.0
    dd['c']['b'] = 2.2
    dd['c']['c'] = 1.0

    dm1 = DistanceMatrix(doubleDict = dd)

    dm2 = copy.deepcopy(dm1)
    dm2['a']['c'] = dm2['c']['a'] = 0.5

    print distanceMatrixCorrelation(dm1, dm2, None)







