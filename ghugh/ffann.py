import math, random, collections, numbers
from . import util

def sigmoid(x):
    if x > 36: x = 36
    elif x < -36: x = -36
    ret = 1.0 / (1.0 + math.exp(-x))
    assert ret not in (0.0, 1.0), "Invalid sigmoid value %f->%f" % (x, ret)
    return ret

def dsigmoid(y):
    ret = y*(1.0 - y)
    assert ret != 0.0, "Invalid sigmoid value %f->%f" % (y, ret)
    return ret

class _Layer(collections.Sequence):
    def __init__(self, count):
        self._count = count
        self._outputs = [0.0] * count

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        return self._outputs[index]

class _OLayer(_Layer):
    def __init__(self, count, ocount, iweights, bias=False):
        self._bias = bias
        if bias: count += 1
        super().__init__(count)
        if iweights is None:
            self._weights = [[random.uniform(-1.0, 1.0)] * count
                                for _ in range(ocount)]
        elif isinstance(iweights, numbers.Number):
            self._weights = [[iweights] * count
                                for _ in range(ocount)]
        elif isinstance(iweights, collections.Sequence):
            if len(iweights) != 2:
                raise ValueError("Two-element sequence is needed for"
                                 "initial weights, got %r" % iweights)
            self._weights = [[random.uniform(iweights[0], iweights[1])] * \
                                   count
                             for _ in range(ocount)]
        elif isinstance(iweights, collections.Iterable):
            iweights = iter(iweights)
            self._weights = [[next(iweights) for _ in range(count)]
                             for _ in range(ocount)]
        elif isinstance(iweights, collections.Callable):
            self._weights = [[iweights() for _ in range(count)]
                             for _ in range(ocount)]
        self._weightsAt = util.transposed(self._weights)

    def bias(self):
        return self._bias

    def weightsTo(self, index):
        return self._weights[index]
    
    def weightsAt(self, index):
        return self._weightsAt[index]

    def fix(self, odeltas, N=0.1):
        if self._bias:
            deltas = [0.0] * (len(self) - 1)
            for oi, od in enumerate(odeltas):
                weights = self._weights[oi]
                for (i, (o, w)) in enumerate(zip(self, weights)):
                    if i!= 0: deltas[i-1] += od * w
                    weights[i] = w + N*od*o
        else:
            deltas = [0.0] * len(self)
            for oi, od in enumerate(odeltas):
                weights = self._weights[oi]
                for (i, (o, w)) in enumerate(zip(self, weights)):
                    deltas[i] += od * w
                    weights[i] = w + N*od*o

        return deltas


class InputLayer(_OLayer):
    def __init__(self,
                 count, ocount,
                 iweights=None,
                 function=None, bias=None):
        super().__init__(count, ocount, iweights, bias=bias is not None)
        if bias is not None: self._outputs[0] = bias
        self._function = function
    
    def activate(self, inputs_):
        inputs = iter(inputs_)
        for i in range(self._bias and 1 or 0, len(self)):
            ii = next(inputs)
            self._outputs[i] = self._function and self._function(ii) or ii
        return self

    def __repr__(self):
        return "input[%d, bias=%r]" % (len(self), self._bias)


class _ILayer(_Layer):
    def __init__(self, count, function=sigmoid, dfunction=dsigmoid):
        super().__init__(count)
        self._function = function
        self._dfunction = dfunction

    def dfunction(self):
        return self._dfunction

    def activate(self, inputs):
        return self._activate(inputs, len(self), 0)

    def _activate(self, inputs, count, shift):
        for o in range(0, count):
            s = sum(i*w
                                                  for i,w in
                                                  zip(inputs,
                                                      inputs.weightsTo(o)))
            self._outputs[o + shift] = self._function(s)
        return self


class HiddenLayer(_OLayer, _ILayer):
    def __init__(self, count, ocount,
                       iweights=None,
                       function=sigmoid, dfunction=dsigmoid, bias=None):
        super().__init__(count, ocount, iweights,
                         bias=bias is not None)
        self._function = function
        self._dfunction = dfunction
        if bias is not None: self._outputs[0] = bias

    def activate(self, inputs):
        count = len(self)
        shift = 0
        if self._bias:
            count -= 1
            shift = 1
        return self._activate(inputs, count, shift)

    def __repr__(self):
        return "hidden[%d, bias=%r]" % (len(self), self._bias)


class OutputLayer(_ILayer):
    def __init__(self, count, function=sigmoid, dfunction=dsigmoid):
        super().__init__(count, function, dfunction)

    def __repr__(self):
        return "output[%d] x->%s" % (len(self), self._function.__name__)
        
class Net(object):
    def __init__(self, *layers):
        super().__init__()
        if not layers or len(layers) < 2:
            raise ValueError("At least two layers needed, got %d" % 
                             len(layers))
        self._layers = tuple(layers)
        self._feed = util.compose(*tuple(l.activate for l in self._layers))

    def feed(self, data):
        return self._feed(data)

    def __repr__(self):
        return " ".join(str(l) for l in self._layers)
