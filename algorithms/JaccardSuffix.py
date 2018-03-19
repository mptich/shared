# Jaccard suffix routines
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

import sys

def JaccardSuffixDistance(suff1, suff2, matrix):
    """
    :param suff1 - first suffix
    :param suff2 - 2nd suffix
    :param matrix - Blosum Matrix
    :return - tuple(r, r1, r2): rate of (suff1, suff2) pair,
    based on the matrix; rate of suff1; rate of suff2
    """
    assert(len(suff1) == len(suff2))
    rate1 = sum([matrix.lookup_score(x,x) for x in suff1])
    rate2 = sum([matrix.lookup_score(x,x) for x in suff2])
    rate = sum([matrix.lookup_score(x,y) for x,y in zip(suff1,suff2)])
    assert(rate1 >= rate)
    assert(rate2 >= rate)
    return (rate, rate1, rate2)

def JaccardSuffixRate(suff1, suff2, matrix):
    """
    :param suff1 - first suffix
    :param suff2 - 2nd suffix
    :matrix - Blosum Matrix
    :return - absolute rate of (suff1, suff2) pair, based on the matrix
    """
    assert(len(suff1) == len(suff2))
    return sum([matrix.lookup_score(x,y) for x,y
                in zip(suff1, suff2)])

def JaccardSuffixStringToSet(s, length):
    ss = set()
    for i in range(len(s) - length + 1):
        ss.add(s[i : i + length])
    return ss

def JacardSuffixCompareStrings(s1, s2, length, matrix):
    """
    :param s1: 1st string
    :param s2: 2nd string
    :param length: suffix length
    :param marix - Blosum Matrix
    :return: (dict1, dict2) - 2 dictionaries mapping each siffix into the
    best suffix (by highest score) in the other string
    """
    dict1 = {}
    dict2 = {}
    ss1 = JaccardSuffixStringToSet(s1, length)
    ss2 = JaccardSuffixStringToSet(s2, length)
    for suff in ss1:
        rate = -sys.maxint
        for suff1 in ss2:
            r = JaccardSuffixRate(suff, suff1, matrix)
            if r > rate:
                rate = r
                dict1[suff] = suff1
    for suff in ss2:
        rate = -sys.maxint
        for suff1 in ss1:
            r = JaccardSuffixRate(suff, suff1, matrix)
            if r > rate:
                rate = r
                dict2[suff] = suff1
    return (dict1, dict2)












