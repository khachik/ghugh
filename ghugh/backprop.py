from trainer import Algo

class BackPropagation(Algo):
    def __init__(self, net, learningRate=0.01, momentum=0.0, batch=False):
        self.lr = learningRate
        self.momentum = momentum
        self.batch = batch
        self.net = net
        self.input = Input(self.net.layers()[0], net.layers()[1], batch)
        self.hiddens = tuple(Hidden(net.layers()[i],
                                    net.layers()[i+1],
                                    batch) for i in
                                               range(1, len(net.layers())))
        self.output = Output(self.net.layers()[-1])

    def train(self, dataset):
        n = 0
        merror = 0.0
        for input, expected in dataset:
            self.net.feed(input)
            merror += self.output.propagate(expected)
            n += 1
            deltas = self.output.deltas()
            for h in self.hidden:
                h.update(deltas, self.lr, self.momentum)
                deltas = h.deltas()
            self.input.update(deltas, self.lr, self.momentum)

        if self.batch:
            for h in self.hidden:
                h.updateWeights()
            self.input.updateWeights()
        return merror/n


class Weighted(object):
    def __init__(self, layer, nextLayer, batch):
        self._oldWDeltas = [[0.0] * len(nextLayer) for _ in layer]
        self._batch = batch
        if batch:
            self._wDeltas = [[0.0] * len(nextLayer) for _ in layer]
    
    def doWeight(self, index, o, odelta, weights, oldDeltas, LR, M):
        delta = LR*o*odelta + M*oldDeltas[index]
        weights[index] += delta
        oldDeltas[index] = delta

    def doWeight_batch(self, index, o, odelta, weights, deltas, LR, M):
        deltas[index] += o*odelta

    def doDelta(self, index, delta):
        raise NotImplemented

    def update(self, odeltas, LR, M):
        dw = self._batch and self.doWeight_batch or self.doWeight
        update(odeltas, self.doDelta, dw, LR, M)

    def _update(odeltas, doDeltas, doWeights, LR, M):
        layer = self._layer
        oldWDeltas = self._oldWDeltas
        for i,o in enumerate(layer):
            weights = layer.weightsAt(i)
            delta = 0.0
            owds = oldWDeltas[i]
            for oi, (od, w) in enumerate(zip(odeltas, weights):
                delta += od*w
                doWeight(oi, o, od, weights, owds, LR, M)
            doDelta(i, o, delta)

    def updateWeights(self, LR, M):
        assert self._batch
        for i,(dws,odws) in enumerate(zip(self._wDeltas, self._oldWDeltas):
            ws = self._layer.weightsAt(i)
            for j, (dw, odw) in enumerate(zip(dws, odws)):
                dw = LR*dw + M*odw
                ws[j] += dw
                odws[j] = dw # FIXME: really dw? Including M*odw?
                dws[j] = 0.0
        

class Input(Weighted):
    def doDelta(self, *args):
        pass

class Hidden(Weighted):
    def __init__(self, layer, nextLayer, batch):
        super().__init__(self, layer, nextLayer, batch)

        # remembet that if the layer has bias, then deltas[0]
        # must be omitted when returning deltas
        # the code below considers that fact, but it is tricky
        df = layer.dfunction()
        if layer.bias():
            self._deltas = [0.0] * (len(layer) - 1)
            # one would think that the following code shouldn't work
            # but it does, since the item at 0 index will be put at the end
            # of the list and then overridden by the last one.
            self.doDelta = lambda index, output, error: \
                               self._deltas[index-1] = df(output)*error
        else:
            self._deltas = [0.0] * len(layer)
            self.doDelta = lambda index, output, error: \
                                self._deltas[index] = df(output)*error

    def deltas(self):
        return self._deltas

class Output(object):
    def __init__(self, layer):
        self._deltas = [0.0] * len(layer)
        self._layer = layer

    def propagate(self, expected):
        df = self._layer.dfunction()
        se = 0.0
        for i,(a,e) in enumerate(zip(self._layer, expected)):
            error = e - a
            self._deltas[i] = df(a)*error
            se += error * error
        return se

    def deltas(self):
        return self._deltas
