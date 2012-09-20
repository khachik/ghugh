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

class WeightedLayerAlgo(object):
    
    def init(self, bias):
        self.bias = bias
        self.ws = [[4, 5, 6],
                   [7, 8, 9]]
        if self.bias:
            self.ws[0].insert(0, 31)
            self.ws[1].insert(0, 23)

        self.deltaWs = [[0.0]*len(self.ws[0]) for _ in range(len(self.ws))]
        self.oldDeltaWs = [[0.0]*len(self.ws[0]) 
                           for _ in range(len(self.ws[0]))]

        self.i = [1, 2, 3]

    def updateAlgo(self, odelta1, odelta2, LR, M):
        self.algo.update([odelta1, odelta2], LR, M)

        # deltaW[j][i] = LR*o[i]*odeltas[j] + M*oldDeltaWij
        s = self.bias and 1 or 0
        self.deltaWs = [[LR*self.i[0]*odelta1 + M*self.oldDeltaWs[s][0],
                         LR*self.i[0]*odelta2 + M*self.oldDeltaWs[s][1]],
                        [LR*self.i[1]*odelta1 + M*self.oldDeltaWs[s+1][0],
                         LR*self.i[1]*odelta2 + M*self.oldDeltaWs[s+1][1]],
                        [LR*self.i[2]*odelta1 + M*self.oldDeltaWs[s+2][0],
                         LR*self.i[2]*odelta2 + M*self.oldDeltaWs[s+2][1]]]
        if self.bias:
            self.deltaWs.insert(0, [LR*1*odelta1 + M*self.oldDeltaWs[0][0],
                                    LR*1*odelta2 + M*self.oldDeltaWs[0][1]])

        # w[j][i] = w[j][i] + deltaW[i][j]
        for j,dws in enumerate(self.deltaWs):
            for i,dw in enumerate(dws):
                self.ws[i][j] += dw

    def testLayerWeights(self):
        self.updateAlgo(11, 12, 100, 1000)
        actual = self.h._weights
        self.assertEqual(self.ws, actual)
    
    def testAlgoDeltaWeights(self):
        self.updateAlgo(11, 12, 100, 1000)
        oldDeltas = self.deltaWs
        self.assertEqual(oldDeltas, self.algo._oldWDeltas) 
    
    def testLayerWeightsTwoPass(self):
        self.updateAlgo(11, 12, 100, 1000)
        self.oldDeltaWs = [list(i) for i in self.deltaWs]
        self.updateAlgo(-13, -14, 100, 1000)
        self.assertEqual(self.ws, self.h._weights)

    def testAlgoDeltaWeightsTwoPass(self):
        self.updateAlgo(11, 12, 100, 1000)
        self.oldDeltaWs = [list(i) for i in self.deltaWs]
        self.updateAlgo(-13, -14, 100, 1000)
        oldDeltas = [list(i) for i in self.deltaWs]
        self.assertEqual(oldDeltas, self.algo._oldWDeltas)

class HiddenAlgo(WeightedLayerAlgo):

    def init(self, bias):
        super().init(bias)
        self.h = LayerMock(self.i, [list(i) for i in self.ws],
                           self.bias, lambda y: y)
        self.algo = Hidden(self.h, 2, False)

class TestHiddenAlgoOnlineNoBias(HiddenAlgo, unittest.TestCase):

    def setUp(self):
        super().init(False)

    def deltas(self, odelta1, odelta2):
        #  delta[i] = sumOf(odelta[j]*ws[j][i]*df(o[i]) j=0..J)
        s = self.bias and 1 or 0
        deltas = [odelta1*self.ws[0][s]*self.i[0] + \
                  odelta2*self.ws[1][s]*self.i[0],
                  odelta1*self.ws[0][s+1]*self.i[1] + \
                  odelta2*self.ws[1][s+1]*self.i[1],
                  odelta1*self.ws[0][s+2]*self.i[2] + \
                  odelta2*self.ws[1][s+2]*self.i[2]]
        return deltas

    def testDeltas(self):
        expected = self.deltas(11, 12)
        self.algo.update([11, 12], 1, 1)
        actual = list(self.algo.deltas())
        self.assertEqual(expected, actual)


class TestHiddenAlgoOnlineBias(TestHiddenAlgoOnlineNoBias):
    def setUp(self):
        self.init(True)

class InputAlgo(WeightedLayerAlgo):

    def init(self, bias):
        super().init(bias)
        self.h = LayerMock(self.i, [list(i) for i in self.ws],
                           self.bias, lambda y: y)
        self.algo = Input(self.h, 2, False)

class TestInputAlgoOnlineNoBias(InputAlgo, unittest.TestCase):

    def setUp(self):
        self.init(False)


class TestInputAlgoOnlineBias(TestInputAlgoOnlineNoBias):
    
    def setUp(self):
        self.init(True)

class WeightedAlgoBatch(object):

    def init(self, bias):
        self.bias = bias
        self.i = [1, 2, 3]
        self.ws = [[4, 5, 6],
                   [7, 8, 9]]
        if self.bias:
            self.ws[0].insert(0, 31)
            self.ws[1].insert(0, 23)

        self.deltaWs = [[0.0]*len(self.ws) for _ in range(len(self.ws[0]))]
        self.accuDeltaWs = \
                [[0.0]*len(self.ws) for _ in range(len(self.ws[0]))]
        self.oldDeltaWs = \
                [[0.0]*len(self.ws) for _ in range(len(self.ws[0]))]

    def updateAlgo(self, odelta1, odelta2, LR, M):
        self.algo.update([odelta1, odelta2], LR, M)

        # deltaW[j][i] = o[i]*odeltas[j] + M*oldDeltaWij
        s = self.bias and 1 or 0
        self.deltaWs = [[self.i[0]*odelta1, self.i[0]*odelta2],
                        [self.i[1]*odelta1, self.i[1]*odelta2],
                        [self.i[2]*odelta1, self.i[2]*odelta2]]
        if self.bias:
            self.deltaWs.insert(0, [1*odelta1, 1*odelta2])

    def updateDeltaWs(self, deltaWs):
        self.accuDeltaWs = [[o+n for o,n in zip(os,ns)]
                            for os,ns in zip(self.accuDeltaWs, deltaWs)]

    def updateWeights(self, deltaWs, LR, M):
        # w[j][i] = w[j][i] + deltaWs[i][j]
        # deltaWs[i][j] = LR*deltaWs[i][j] + M*oldDeltaWs[i][j]
        for j,dws in enumerate(deltaWs):
            for i,dw in enumerate(dws):
                dw = LR*dw + M*self.oldDeltaWs[j][i]
                self.ws[i][j] += dw
                dws[i] = 0.0
                self.oldDeltaWs[j][i] = dw

    def testAccuDeltaWsOnePass(self):
        self.updateAlgo(17, 23, 100, 1000)
        self.updateDeltaWs(self.deltaWs)
        expected = self.deltaWs
        actual = self.algo._wDeltas
        self.assertEqual(expected, actual)

    def testAccuDeltaWsTwoPass(self):
        self.testAccuDeltaWsOnePass()
        self.updateAlgo(31, 37, 100, 1000)
        self.updateDeltaWs(self.deltaWs)
        expected = self.accuDeltaWs
        actual = self.algo._wDeltas
        self.assertEqual(expected, actual)

    def testAccuDeltaWsThreePass(self):
        self.testAccuDeltaWsTwoPass()
        self.updateAlgo(41, 43, 100, 1000)
        self.updateDeltaWs(self.deltaWs)
        expected = self.accuDeltaWs
        actual = self.algo._wDeltas
        self.assertEqual(expected, actual)

    def testLayerWeights(self):
        self.testAccuDeltaWsThreePass()
        self.algo.updateWeights(100, 1000)
        self.updateWeights(self.accuDeltaWs, 100, 1000)
        expected = self.ws
        actual = self.h._weights
        self.assertEqual(expected, actual)

    def testOldDeltasOneEpoch(self):
        self.testLayerWeights()
        expected = self.oldDeltaWs
        actual = self.algo._oldWDeltas
        self.assertEqual(expected, actual)
        

    def deltaTest(self, od1, od2):
        pass

    def test2ndEpoch(self):
        self.testOldDeltasOneEpoch()

        for odeltas in ((47, 49), (53, 59), (61, 67)):
            self.updateAlgo(odeltas[0], odeltas[1], 1, 1)
            self.deltaTest(odeltas[0], odeltas[1])
            self.updateDeltaWs(self.deltaWs)
            expected = self.accuDeltaWs
            actual = self.algo._wDeltas
            self.assertEqual(expected, actual)
        
        self.algo.updateWeights(100, 1000)
        self.updateWeights(self.accuDeltaWs, 100, 1000)
        expected = self.ws
        actual = self.h._weights
        self.assertEqual(expected, actual)

        expected = self.oldDeltaWs
        actual = self.algo._oldWDeltas
        self.assertEqual(expected, actual)


class HiddenAlgoBatch(WeightedAlgoBatch):
    def init(self, bias):
        super().init(bias)
        self.h = LayerMock(self.i, [list(i) for i in self.ws],
                           self.bias, lambda y: y)
        self.algo = Hidden(self.h, 2, True)

    def deltas(self, odelta1, odelta2):
        s = self.bias and 1 or 0
        deltas = [odelta1*self.ws[0][s]*self.i[0] + \
                  odelta2*self.ws[1][s]*self.i[0],
                  odelta1*self.ws[0][s+1]*self.i[1] + \
                  odelta2*self.ws[1][s+1]*self.i[1],
                  odelta1*self.ws[0][s+2]*self.i[2] + \
                  odelta2*self.ws[1][s+2]*self.i[2]]
        return deltas

    def deltaTest(self, od1, od2):
        expected = self.deltas(od1, od2)
        actual = self.algo.deltas()
        self.assertEqual(expected, actual)

    def testDeltas(self):
        self.algo.update([17, 23], 1, 1)
        self.deltaTest(17, 23)
        

class TestHiddenAlgoBatchNoBias(HiddenAlgoBatch, unittest.TestCase):
    def setUp(self):
        super().init(False)


class TestHiddenAlgoBatchBias(TestHiddenAlgoBatchNoBias):
    def setUp(self):
        self.init(True)


class InputAlgoBatch(WeightedAlgoBatch):

    def init(self, bias):
        super().init(bias)
        self.h = LayerMock(self.i, [list(i) for i in self.ws],
                           self.bias, lambda y: y)
        self.algo = Input(self.h, 2, True)


class TestInputAlgoBatchNoBias(InputAlgoBatch, unittest.TestCase):

    def setUp(self):
        super().init(False)


class TestInputAlgoBatchBias(TestInputAlgoBatchNoBias):

    def setUp(self):
        self.init(True)

if __name__ == "__main__":
    unittest.main()
