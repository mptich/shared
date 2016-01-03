# This module implements a Weighted Kendall Tau Distance algorithm,
# described in the article:

import numpy as np

# list1 - one of the input lists
# list2 - the 2nd input list
# weights - list of weights
def calculateWeighetedKendall(list1, list2, weights):
    dist = 0.0
    length = len(list1)
    if length < 2:
        return 1.
    assert(length == len(list2))
    assert(length == len(weights))

    dstList = []
    for i in range(length):
        dstList.append((list1[i], list2[i], weights[i]))
    # First, sort the 2nd list, then 1st, to have items from teh 2nd list in
    # the right order in case of equal values of the 1st list
    dstList = sorted(dstList, key = lambda x: x[1])
    dstList = sorted(dstList, key = lambda x: x[0])

    mergeLen = 1
    while mergeLen < length:
        srcList = np.array(dstList)
        dstList = []
        startPos = 0
        while startPos < length - mergeLen:
            dist = mergeSublists(srcList[startPos : startPos + mergeLen],
                srcList[startPos + mergeLen : startPos + (mergeLen << 1)],
                dstList, dist)
            startPos += (mergeLen << 1)
        # Add the rest
        dstList += list(srcList[startPos:])
        mergeLen <<= 1

    return 1. - 2 * dist / sum(weights) / (length - 1)

# Merge sort with summarazing weights
def mergeSublists(src1, src2, dst, dist):
    length1 = len(src1)
    length2 = len(src2)
    assert(length2 <= length1)
    sum = 0.
    integWeights = []
    for i in range(length1):
        sum += src1[length1 - 1 - i][2]
        integWeights.append(sum)

    pos1 = 0
    pos2 = 0
    while (pos1 < length1) and (pos2 < length2):
        if src1[pos1][1] <= src2[pos2][1]:
            dst.append(src1[pos1])
            pos1 += 1
        else:
            dst.append(src2[pos2])
            move = length1 - pos1
            dist += integWeights[move - 1] + src2[pos2][2] * move
            pos2 += 1

    if pos1 == length1:
        dst += list(src2[pos2:])
    else:
        dst += list(src1[pos1:])

    return dist

# Test
if __name__ == "__main__":
    print calculateWeighetedKendall([1,2,3,4], [4,3,2,1], [0.6, 0.8, 1, 0.3])
    print calculateWeighetedKendall([1,2,3,4,5], [1,2,3,4,5], [0.6, 0.8, 1,
        0.3, 0.8])
    print calculateWeighetedKendall([1,2,3,4,5], [1,4,3,2,5], [0.6, 100, 100,
        100, 0.8])
    print calculateWeighetedKendall([1,2,3,4,5], [1,4,3,2,5], [0.6, 1, 1,
        1, 0.8])
    print calculateWeighetedKendall([1,2,3,4,5], [1,4,3,2,5], [0.6, 0.1, 0.1,
        0.1, 0.8])
    print calculateWeighetedKendall([1,2,3,4,5,6], [1,2,4,6,5,3], [1, 0.6,
        0.4, 0.3, 0.2, 0.1])
    print calculateWeighetedKendall([1,2,3,4,5,6], [1,2,4,6,5,3], [1, 1,
        1, 1, 1, 1])



