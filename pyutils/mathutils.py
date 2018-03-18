# Shared utilities and classes
#
# Copyright (C) 20015-2018  Author: Misha Orel
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


from shared.pyutils.utils import *
from scipy.fftpack import fft as FftTransform

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


@UtilStaticVars(cachedBlackman = {})
def UtilCalculateFft(data, axis=0):
    shape = data.shape
    assert shape[-1] == 2

    if len(shape) > 2:
        data = data.reshape((-1, 2))
    data = data[:, 0] + data[:, 1] * 1j
    data = data.reshape(shape[:-1])

    axisLen = data.shape[axis]
    blackman = UtilCalculateFft.cachedBlackman.get(axisLen, None)
    if blackman is None:
        blackman = np.blackman(axisLen)
        UtilCalculateFft.cachedBlackman[axisLen] = blackman

    bmShape = tuple()
    for i in range(len(data.shape)):
        if i == axis:
            bmShape += (axisLen,)
        else:
            bmShape += (1,)
    blackman = blackman.reshape(bmShape)
    data *= blackman

    return FftTransform(data, axis=axis)