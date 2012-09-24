from abc import abstractmethod

from .trainer import Algo

class Backpropagation(Algo):
    """Backpropagation algorithm using gradient descent.
    Instances of this class are stateful."""

    def __init__(self, net, batch=False):
        """Initializes an algorithm instance for training the 
        given net instance. If batch is True the full dataset 
        is applied before adjusting the weights in the net
        as opposed to the online training mode (batch=False)
        when weights are updated after each training data."""

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
        """Trains the net against the given dataset
        (a sequence of 2-element tuples) with LR
        learning rate and M momentum.
        Returns the average error."""

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
        """backpropagation for a single data.
        Returns the squere error."""

        self.net.feed(input)
        error = self.output.propagate(expected)
        deltas = self.output.deltas()
        for h in self.hiddens:
            h.update(deltas, LR, M)
            deltas = h.deltas()
        self.input.update(deltas, LR, M)
        return error
        
        
class Weighted(object):
    """Backpropagation for a weighted layer.
       For batch training, updateWeights method 
       must be called to apply weights changes.
       Instances of this class are stateful.
    """

    def __init__(self, layer, cNextLayer, batch):
        """Initializes a trainer for the given layer.
        cNextLayer is the number of neurons for the next
        layer in the net. batch is a boolean flag indicating
        batch training mode."""

        self._layer = layer
        self._oldWDeltas = [[0.0] * cNextLayer for _ in layer]
        self._batch = batch
        if batch:
            self._wDeltas = [[0.0] * cNextLayer for _ in layer]
    
    def doWeights(self, index, o, odelta, weights, oldDeltas, LR, M):
        """Updates the weight by the index according to the
        following formulae:
            deltaW[i][j] = LR*output[i]*output_delta[j] +
                           M*oldDeltaW[i][j]
            w[i][j] = w[i][j] + deltaW[i][j]
            oldDeltaW[i][j] = deltaW[i][j]
        where:
            * deltaW[i][j] - delta weight for the connection between
                             the ith neuron in the current layer and
                             the jth neuron in the next layer;
            * output[j] - output signal of the ith neuron in the current
                          layer (o argument);
            * output_delta[j] - is the backpropagated error at the jth
                                neuron in the next layer
                                (odelta argument);
            * oldDeltaW[i] - deltaW[i][j] calculated in the previous
                             iteration initially 0
                             (items in oldDeltas argument);
            * LR - learning rate;
            * M - momentum.
        """

        delta = LR*o*odelta + M*oldDeltas[index]
        weights[index] += delta
        oldDeltas[index] = delta

    def doWeights_batch(self, index, o, odelta, weights, deltas, LR, M):
        """Accumulates the current error (o*odelta) for further weight
        update calculations."""

        deltas[index] += o*odelta

    @abstractmethod
    def doDeltas(self, index, delta):
        """Stores the backpropagated error for the 
        neuron at the given index."""
        raise NotImplementedError

    def update(self, odeltas, LR, M):
        """Backpropagates the errors from the next layer, calculates
        and stores errors for the current layer. In online training mode,
        updates layer weights. For batch mode, accumulates weight
        changes to apply after the full dataset is exhausted."""

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
        """Updates weights in batch training mode. Must be called
        when the full dataset is examined by `update'"""
        assert self._batch
        for i,(dws,odws) in enumerate(zip(self._wDeltas, self._oldWDeltas)):
            ws = self._layer.weightsAt(i)
            for j, (dw, odw) in enumerate(zip(dws, odws)):
                dw = LR*dw + M*odw
                ws[j] += dw
                odws[j] = dw
                dws[j] = 0.0
        

class Input(Weighted):
    """Backpropagation for the input layer."""

    def doDeltas(self, *args):
        # no need to collect deltas
        pass

class Hidden(Weighted):
    """Backpropagation for hidden layers."""

    def __init__(self, layer, cNextLayer, batch):
        super().__init__(layer, cNextLayer, batch)

        # remembet that if the layer has bias, then deltas[0]
        # must be omitted when returning deltas
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
    """Backpropagation for output layer."""

    def __init__(self, layer):
        self._deltas = [0.0] * len(layer)
        self._layer = layer

    def propagate(self, expected):
        """Calculates output deltas for each
        neuron in output layer and returns 1/2 of
        the squere error."""

        df = self._layer.dfunction()
        se = 0.0
        for i,(a,e) in enumerate(zip(self._layer, expected)):
            error = e - a
            self._deltas[i] = df(a)*error
            se += error * error
        return se/2.0

    def deltas(self):
        return self._deltas
