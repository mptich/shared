# This module implements a Weighted Kendall Tau Distance algorithm,
# described in the article:

import numpy as np

# list1 - one of the input lists
# list2 - the 2nd input list
# weights - list of weights
# components - if not None (should be a dictionary), then it will return
# how much each of unique elements of list1 contributed in weights
# reordering
def calculateWeightedKendall(list1, list2, weights=None, components=None):
    dist = 0.0
    length = len(list1)
    if weights is None:
        weights = [1.0] * length
    if length < 2:
        return 1.
    assert(length == len(list2))
    assert(length == len(weights))

    # First, sort the 2nd list, then 1st, to have items from teh 2nd list in
    # the right order in case of equal values of the 1st list
    dstList = zip(list1, list2, weights)
    dstList = sorted(dstList, key = lambda x: x[1])
    dstList = sorted(dstList, key = lambda x: x[0])
    # Now add ordinal to each tuple: unzip it, and zip together with ordinals
    dstList = zip(*(zip(*dstList) + [range(length)]))

    # Calculate the total weight of transactions in teh worst case
    # It is sum(weights) * (length - 1), minus all possible transactions
    # within the intervals where values of list1 are equal
    worstCaseWeight = sum(weights) * (length - 1)
    prevValue = None
    eqCount = 1
    accumWeight = 0.
    for curValue, _, weight, _ in (dstList + [(None, None, None, None)]):
        if curValue == prevValue:
            eqCount += 1
            accumWeight += weight
        else:
            if eqCount > 1:
                worstCaseWeight -= accumWeight * (eqCount - 1)
            eqCount = 1
            accumWeight = weight
            prevValue = curValue
    if worstCaseWeight == 0.:
        # All values in list1 are equal
        return 0.

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

    if components is not None:
        for ind, el in enumerate(dstList):
            pos = el[3]
            # Add change of position multiplied by weight
            components[el[0]] = components.get(el[0], 0.) + \
                el[2] * abs(ind - pos)
        # Normalize components values
        s = sum(components.values())
        for k, v in components.items():
            if s != 0.:
                components[k] = v / s
            else:
                components[k] = 1. / len(components)

    return 1. - 2 * dist / worstCaseWeight

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
            assert(src1[pos1][0] < src2[pos2][0])
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
    print calculateWeightedKendall([1,2,3,4], [4,3,2,1], [0.6, 0.8, 1, 0.3])
    print calculateWeightedKendall([1,2,3,4,5], [1,2,3,4,5], [0.6, 0.8, 1,
        0.3, 0.8])
    print calculateWeightedKendall([1,2,3,4,5], [1,4,3,2,5], [0.6, 100, 100,
        100, 0.8])
    print calculateWeightedKendall([1,2,3,4,5], [1,4,3,2,5], [0.6, 1, 1,
        1, 0.8])
    print calculateWeightedKendall([1,2,3,4,5], [1,4,3,2,5], [0.6, 0.1, 0.1,
        0.1, 0.8])
    print calculateWeightedKendall([1,2,3,4,5,6], [1,2,4,6,5,3], [1, 0.6,
        0.4, 0.3, 0.2, 0.1])
    print calculateWeightedKendall([1,2,3,4,5,6], [1,2,4,6,5,3], [1, 1,
        1, 1, 1, 1])
    print calculateWeightedKendall([1,2,3,4,5,6], [6,5,2,4,3,1], [1, 1,
        1, 1, 1, 1])
    components = {}
    print calculateWeightedKendall([1,2,2,1,2,1], [4,1,2,5,3,6], [0.6, 0.5,
        1.7, 10., 4.3, 0.8])
    print "Comp ", components
    components = {}
    print calculateWeightedKendall([1,2,2,1,2,1], [3,6,5,2,4,1], [0.6, 0.5,
        1.7, 10., 4.3, 0.8], components)
    print "Comp ", components
    components = {}
    print calculateWeightedKendall([1,2,2,1,2,1], [4,6,5,2,3,1], None,
                                   components)
    print "Comp ", components



