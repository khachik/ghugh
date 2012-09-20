from .trainer import Algo

class BackPropagation(Algo):
    def __init__(self, net, batch=False):
        super().__init__()
        self.batch = batch
        self.net = net
        self.input = Input(self.net.layers()[0],
                           self.net.layers()[1].inputSize(), batch)
        self.hiddens = []
        for i in range(1, len(net.layers())-1):
            nextLen = net.layers()[i+1].inputSize()
            self.hiddens.append(Hidden(net.layers()[i], nextLen, batch))
        self.output = Output(self.net.layers()[-1])

    def train(self, dataset, LR, M):
        n = 0
        merror = 0.0
        for input, expected in dataset:
            merror += self.propagate(input, expected, LR, M)
            n += 1
        if self.batch:
            for h in self.hiddens:
                h.updateWeights(LR, M)
            self.input.updateWeights(LR, M)
        return merror/n

    def propagate(self, input, expected, LR, M):
        self.net.feed(input)
        error = self.output.propagate(expected)
        deltas = self.output.deltas()
        for h in self.hiddens:
            h.update(deltas, LR, M)
            deltas = h.deltas()
        self.input.update(deltas, LR, M)
        return error
        
        
class Weighted(object):
    def __init__(self, layer, cNextLayer, batch):
        self._layer = layer
        self._oldWDeltas = [[0.0] * cNextLayer for _ in layer]
        self._batch = batch
        if batch:
            self._wDeltas = [[0.0] * cNextLayer for _ in layer]
    
    def doWeights(self, index, o, odelta, weights, oldDeltas, LR, M):
        delta = LR*o*odelta + M*oldDeltas[index]
        weights[index] += delta
        oldDeltas[index] = delta

    def doWeights_batch(self, index, o, odelta, weights, deltas, LR, M):
        deltas[index] += o*odelta

    def doDeltas(self, index, delta):
        raise NotImplementedError

    def update(self, odeltas, LR, M):
        dws = self._batch and self.doWeights_batch or self.doWeights
        oldWDeltas = self._batch and self._wDeltas or self._oldWDeltas
        self._update(odeltas, oldWDeltas, self.doDeltas, dws, LR, M)

    def _update(self, odeltas, oldWDeltas, doDeltas, doWeights, LR, M):
        layer = self._layer
        for i,(o, owds) in enumerate(zip(layer, oldWDeltas)):
            weights = layer.weightsAt(i)
            delta = 0.0
            for oi, (od, w) in enumerate(zip(odeltas, weights)):
                delta += od*w
                doWeights(oi, o, od, weights, owds, LR, M)
            doDeltas(i, o, delta)

    def updateWeights(self, LR, M):
        assert self._batch
        for i,(dws,odws) in enumerate(zip(self._wDeltas, self._oldWDeltas)):
            ws = self._layer.weightsAt(i)
            for j, (dw, odw) in enumerate(zip(dws, odws)):
                dw = LR*dw + M*odw
                ws[j] += dw
                odws[j] = dw
                dws[j] = 0.0
        

class Input(Weighted):
    def doDeltas(self, *args):
        pass

class Hidden(Weighted):
    def __init__(self, layer, cNextLayer, batch):
        super().__init__(layer, cNextLayer, batch)

        # remembet that if the layer has bias, then deltas[0]
        # must be omitted when returning deltas
        # the code below considers that fact, but it is tricky
        df = layer.dfunction()
        if layer.bias():
            self._deltas = [0.0] * (len(layer) - 1)
            def doDeltas(index, output, error):
                if index > 0:
                    self._deltas[index-1] = df(output) * error
            self.doDeltas = doDeltas
        else:
            self._deltas = [0.0] * len(layer)
            def doDeltas(index, output, error):
                self._deltas[index] = df(output)*error
            self.doDeltas = doDeltas

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
        return se/2.0

    def deltas(self):
        return self._deltas
