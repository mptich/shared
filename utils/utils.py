# Shared utilities and classes

import json

UtilObjectKey = "__utilobjectkey__"

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
        else:
            return json.JSONEncoder.default(self, obj)

def UtilJSONDecoderDictToObj(d):
    if UtilObjectKey in d:
        moduleName, _, className = d[UtilObjectKey].rpartition('.')
        assert(moduleName)
        module = __import__(moduleName)
        classType = getattr(module, className)
        kwargs = dict((x.encode('ascii'), y) for x, y in d.items())
        inst = classType(**kwargs)
    else:
        inst = d
    return inst

