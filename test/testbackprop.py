import math, unittest, collections
from ghugh import util

from ghugh.ffann import *
from ghugh.backprop import *

class Layer(collections.Sequence):
    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs
    
    def __len__(self):
        return len(self._outputs)

    def __getitem__(self, index):
        return self._outputs[index]


class Weighted(Layer):
    def __init__(self, outputs, weights, bias):
        if bias: super().__init__([1] + outputs)
        else: super().__init__(outputs)
        self._weights = weights
        self.bias = lambda: bias
        self._weightsAt = util.transposed(self._weights)

    def weightsTo(self, index):
        return self._weights[index]

    def weightsAt(self, index):
        return self._weightsAt[index]


class LayerMock(Weighted):
    def __init__(self, outputs, weights, bias, df):
        super().__init__(outputs, weights, bias)
        self.dfunction = lambda: df
        

class OLayerMock(Layer):
    def __init__(self, outputs, df):
        super().__init__(outputs)
        self.dfunction = lambda: df


class TestOutputDeltas(unittest.TestCase):
    def setUp(self):
        self.o = OLayerMock((1, 2), lambda y: y)
        self.algo = Output(self.o)

    def testDeltas(self):
        error = self.algo.propagate((3, 6))
        self.assertEqual(2.0+8.0, error)
        deltas = list(self.algo.deltas())
        self.assertEqual(2, len(deltas))
        self.assertEqual(1*2, deltas[0])
        self.assertEqual(2*4, deltas[1])

class TestHiddenOnlineNoBias(unittest.TestCase):

    def setUp(self):
        self.ws = [[4, 5, 6],
                   [7, 8, 9]]
        self.deltaWs = [[0.0]*len(self.ws[0]) for _ in range(len(self.ws))]
        self.oldDeltaWs = [[0.0]*len(self.ws[0]) 
                           for _ in range(len(self.ws[0]))]

        self.h = LayerMock([1, 2, 3], [list(i) for i in self.ws],
                           False, lambda y: y)
        self.algo = Hidden(self.h, 2, False)

    def updateAlgo(self, odelta1, odelta2, LR, M):
        self.algo.update([odelta1, odelta2], LR, M)

        # i = 0, 1, 2
        # j = 0, 1

        #  delta[i] = sumOf(odelta[j]*ws[j][i]*df(o[i]) j=0..J)
        deltas = [odelta1*self.ws[0][0]*1 + odelta2*self.ws[1][0]*1,
                  odelta1*self.ws[0][1]*2 + odelta2*self.ws[1][1]*2,
                  odelta1*self.ws[0][2]*3 + odelta2*self.ws[1][2]*3]

        # deltaW[j][i] = LR*o[i]*odeltas[j] + M*oldDeltaWij
        self.deltaWs = [[LR*1*odelta1 + M*self.oldDeltaWs[0][0],
                         LR*2*odelta1 + M*self.oldDeltaWs[1][0],
                         LR*3*odelta1 + M*self.oldDeltaWs[2][0]],
                        [LR*1*odelta2 + M*self.oldDeltaWs[0][1],
                         LR*2*odelta2 + M*self.oldDeltaWs[1][1],
                         LR*3*odelta2 + M*self.oldDeltaWs[2][1]]]

        # w[j][i] = w[j][i] + deltaW[j][i]
        for j,dws in enumerate(self.deltaWs):
            for i,dw in enumerate(dws):
                self.ws[j][i] += dw

        return deltas

    def testDeltas(self):
        expected = self.updateAlgo(11, 12, 0.5, 0.2)
        actual = list(self.algo.deltas())
        self.assertEqual(expected, actual)

    def testLayerWeights(self):
        #                odeltas,  LR,  M - Momentum
        self.updateAlgo(11, 12, 100, 1000)
        actual = self.h._weights
        self.assertEqual(self.ws, actual)
    
    def testAlgoDeltaWeights(self):
        self.updateAlgo(11, 12, 100, 1000)
        oldDeltas = [list(i) for i in zip(*self.deltaWs)] # transposed
        self.assertEqual(oldDeltas,
                         self.algo._oldWDeltas) 
    
    def testLayerWeightsTwoPass(self):
        self.updateAlgo(11, 12, 100, 1000)
        self.oldDeltaWs = [list(i) for i in zip(*self.deltaWs)]
        self.updateAlgo(-13, -14, 100, 1000)
        self.assertEqual(self.ws, self.h._weights)

    def testAlgoDeltaWeightsTwoPass(self):
        self.updateAlgo(11, 12, 100, 1000)
        self.oldDeltaWs = [list(i) for i in zip(*self.deltaWs)]
        self.updateAlgo(-13, -14, 100, 1000)
        oldDeltas = [list(i) for i in zip(*self.deltaWs)]
        self.assertEqual(oldDeltas, self.algo._oldWDeltas)

if __name__ == "__main__":
    unittest.main()
