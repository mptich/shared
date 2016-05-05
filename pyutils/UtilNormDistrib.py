from utils import *
import scipy.stats as stats

Norm = stats.norm

class UtilNormDistrib(UtilObject):
    """
    Describes characteristics of normally distributed variable.
    Compatible with UtilObject JSON interface
    """

    def __init__(self, **kwargs):
        if self.buildFromDict(kwargs):
            return
        valList = kwargs.get("list", None)
        if valList:
            self.mean = np.mean(valList)
            if len(valList) >= 2:
                self.std = np.std(valList, ddof = 1.)
            else:
                self.std = None
            self.count = len(valList)
            return
        self.__dict__.update(kwargs)

    def combine(self, other):
        count = self.count + other.count
        mean = (self.mean * self.count + other.mean * other.count) / count
        if self.std is not None:
            selfStd = (self.std * self.std) * (self.count - 1)
        else:
            selfStd = 0.
        if other.std is not None:
            otherStd = (other.std * other.std) * (other.count - 1)
        else:
            otherStd = 0.
        selfDelta = self.mean - mean
        otherDelta = other.mean - mean
        selfDelta = selfDelta * selfDelta * self.count
        otherDelta = otherDelta * otherDelta * other.count
        return UtilNormDistrib(mean = mean, count = count,
            std = math.sqrt((selfStd + otherStd + selfDelta + otherDelta) / \
            (count - 1)))

    def utilJsonDump(self):
        return repr(self)

    def __repr__(self):
        # repr(self.std) because it could be None
        return "<mean: %f std: %s cnt: %u>" % (self.mean, repr(self.std),
            self.count)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        raise AttributeError("%s is not hashable" %
            self.__class__.__name__)

    def less(self, other):
        """
        Probability that this variable is less than the other
        :param other:
        :return:
        """
        if self.std is not None:
            selfStd = self.std
        else:
            selfStd = 0.
        if other.std is not None:
            otherStd = other.std
        else:
            otherStd = 0.
        std = math.sqrt(selfStd * selfStd + otherStd * otherStd)
        mean = other.mean - self.mean
        return (1. - Norm.cdf(0., mean, std))

    @staticmethod
    def minProb(normDistList):
        """
        :param normDistList: List of UtilNormDistrib's
        :return: list of probabilities that a corresponding
            UtilNormDistrib is the minimum
        """
        output = []
        for ord, d in enumerate(normDistList):
            prob = 1.
            for ordd, dd in enumerate(normDistList):
                if ord != ordd:
                    prob *= d.less(dd)
            output.append(prob)
        return output



