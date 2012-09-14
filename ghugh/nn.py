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

class Layer(collections.Sequence):
    def __init__(self, count):
        self._count = count
        self._outputs = [0.0] * count

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        return self._outputs[index]

class OLayer(Layer):
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
            if len(iweights != 2):
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
        self._weightsAt = utiltransposed(self._weights)

    def weightsTo(self, index):
        return self._weights[index]
    
    def weightsAt(self, index):
        


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


class InputLayer(OLayer):
    def __init__(self, count, ocount, function=None, bias=False):
        super().__init__(count, ocount, bias=bias)
        if bias: self._outputs[0] = 1.0
        self._function = function
    
    def activate(self, inputs_):
        inputs = iter(inputs_)
        for i in range(self._bias and 1 or 0, len(self)):
            ii = next(inputs)
            self._outputs[i] = self._function and self._function(ii) or ii
        return self

    def __repr__(self):
        return "input[%d, bias=%r]" % (len(self), self._bias)


class ILayer(Layer):
    def __init__(self, count, function=sigmoid, dfunction=dsigmoid):
        super().__init__(count)
        self._function = function
        self._dfunction = dfunction

    def activate(self, inputs):
        return self._activate(inputs, 0, len(self))

    def _activate(self, inputs, start, end):
        for o in range(start, end):
            s = sum(i*w
                                                  for i,w in
                                                  zip(inputs,
                                                      inputs.weights(o)))
            self._outputs[o] = self._function(s)
        return self


class HiddenLayer(OLayer, ILayer):
    def __init__(self, count, ocount,
                       function=sigmoid, dfunction=dsigmoid, bias=False):
        OLayer.__init__(self, count, ocount, bias=bias)
        ILayer.__init__(self, count, function, dfunction)
        if bias: self._outputs[0] = 0.9

    def fix(self, odeltas, N=0.1):
        deltas = OLayer.fix(self, odeltas, N)
        deltas = iter(deltas)
        return map(lambda x: next(deltas)*self._dfunction(x), self)

    def activate(self, inputs):
        return self._activate(inputs, self._bias and 1 or 0, len(self))

    def __repr__(self):
        return "hidden[%d, bias=%r]" % (len(self), self._bias)


class OutputLayer(ILayer):
    def __init__(self, count, function=sigmoid, dfunction=dsigmoid):
        super().__init__(count, function, dfunction)

    def fix(self, expected):
        e = iter(expected)
        ret = [self._dfunction(o)*(next(e) - o) for o in self]
        return ret

    def __repr__(self):
        return "output[%d] x->%s" % (len(self), self._function.__name__)
        
class Net:
    def __init__(self, *layers, bias=False):
        if not layers or len(layers) < 2:
            raise ValueError("At least two layers needed, got %d" % 
                             len(layers))
        self._layers = []
        self._layers.append(InputLayer(layers[0], layers[1], bias=bias))
        for i in range(1, len(layers)-1):
            self._layers.append(HiddenLayer(layers[i], layers[i+1], bias=bias))
        o = OutputLayer(layers[-1])
        self._layers.append(o)
        self._feed = util.compose(*tuple(l.activate for l in self._layers))
        def fc(layer):
            def f(x, N):
                return layer.fix(x, N), N
            return f
        self._bp = util.compose(
                *tuple(fc(self._layers[i])
                       for i in range(len(self._layers)-2, -1, -1)),
                           unpack=True)

    def feed(self, data):
        return self._feed(data)

    def learn(self, input, output, N=0.1):
        self.feed(input)
        self._bp(self._layers[-1].fix(output), N)

    def __repr__(self):
        return " ".join(str(l) for l in self._layers)

    def diff(self, net2):
        # assuming the structures are identical
        d = {}
        for li, l1 in enumerate(self._layers[:-1]):
            l2 = net2._layers[li]
            ld = {}
            for wsi, ws1 in enumerate(l1._weights):
                wd = {}
                ws2 = l2._weights[wsi]
                for wi, w1 in enumerate(ws1):
                    w2 = ws2[wi]
                    if w1 != w2:
                        wd[wi] = (w1, w2)
                if wd:
                    ld[wsi] = wd

            if ld:
                d[li] = ld
        return d



