import math, unittest, collections

from ghugh.ffann import *
from ghugh.util import transposed

class Weighted(collections.Sequence):
    def __init__(self, inputs, weights):
        self._weights = weights
        self._inputs=  inputs
        self._weightsAt = transposed(self._weights)

    def weightsTo(self, index):
        return self._weights[index]

    def weightsAt(self, index):
        return self._weightsAt[index]

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, index):
        return self._inputs[index]


class TestOutputLayer(unittest.TestCase):

    def setUp(self):
        self.o = OutputLayer(3,
                             function=lambda x: 1000*x + 100000, 
                             dfunction=lambda x: 1000)
        self.inputs = [1, 2]
        self.weights = [[4, 5], [6, 7], [8, 9]]
        self.data = Weighted(self.inputs, self.weights)
        self.response = list(self.o.activate(self.data))

    def testDataSize(self):
        self.assertEqual(3, len(self.o))
        self.assertEqual(3, len(self.response))

    def testActivate(self):
        self.assertEqual((4*1 + 5*2)*1000 + 100000, self.response[0])
        self.assertEqual((6*1 + 7*2)*1000 + 100000, self.response[1])
        self.assertEqual((8*1 + 9*2)*1000 + 100000,
                         self.response[2])


class TestInputLayer(unittest.TestCase):

    def setUp(self):
        self.i = InputLayer(3, 4,
                            iweights=iter([1, 2, 3,
                                           4, 5, 6,
                                           7, 8, 9,
                                           10, 11, 12]))
        self.response = list(self.i.activate([10, 20, 30]))
        self.ib = InputLayer(2, 3, 
                             iweights=iter([1, 2, 3, # bias weights
                                            4, 5, 6,
                                            7, 8, 9]),
                             bias=1)
        self.responseb = list(self.ib.activate([10, 20]))

    def testDataSize(self):
        self.assertEqual(3, len(self.i))
        self.assertEqual(3, len(self.response))
        self.assertEqual(3, len(self.ib))
        self.assertEqual(3, len(self.responseb))

    def testActivate(self):
        self.assertEqual(10, self.response[0])
        self.assertEqual(20, self.response[1])
        self.assertEqual(30, self.response[2])

    def testActivateB(self):
        self.assertEqual(1,  self.responseb[0])
        self.assertEqual(10, self.responseb[1])
        self.assertEqual(20, self.responseb[2])

class TestHiddenLayer(unittest.TestCase):

    def setUp(self):
        self.h = HiddenLayer(3, 2,
                             iweights=iter([1, 2, 3,
                                            4, 5, 6]),
                             function = lambda x: 11*x + 12,
                             dfunction = lambda x: 11)
        self.weighted = Weighted([10, 20, 30, 40],
                                 [[100, 200, 300, 400],
                                  [500, 600, 700, 800],
                                  [900, 1000, 1100, 1200]])
        self.response = list(self.h.activate(self.weighted))

        self.hb = HiddenLayer(2, 1,
                              iweights=iter([1, 2, 3]),
                              function=lambda x: 4*x+5,
                              dfunction=lambda x: 4,
                              bias=1)
        self.weightedb = Weighted([6, 7],
                                  [[8, 9],
                                   [10, 11]])
        self.responseb = list(self.hb.activate(self.weightedb))


    def testDataSize(self):
        self.assertEqual(3, len(self.h))
        self.assertEqual(3, len(self.response))
        self.assertEqual(3, len(self.hb))
        self.assertEqual(3, len(self.responseb))

    def testActivate(self):
        self.assertEqual(11*(10*100 + 20*200 + 30*300 + 40*400) + 12,
                         self.response[0])
        self.assertEqual(11*(10*500 + 20*600 + 30*700 + 40*800) + 12,
                         self.response[1])
        self.assertEqual(11*(10*900 + 20*1000 + 30*1100 + 40*1200) + 12,
                         self.response[2])

    def testActivateB(self):
        self.assertEqual(1, self.responseb[0])
        self.assertEqual(4*(6*8 + 7*9)+5, self.responseb[1])
        self.assertEqual(4*(6*10 + 7*11)+5, self.responseb[2])


class TestNetwork(unittest.TestCase):
    def setUp(self):
        i = InputLayer(2, 3, iweights=iter(range(1, 10)), bias=1)
        h = HiddenLayer(3, 1, iweights=iter(range(10, 14)),
                              function=lambda x: x, bias=1)
        o = OutputLayer(1, function=lambda x: x)
        self.netb = Net(i, h, o)

    def testBiased(self):
        output = list(self.netb.feed((100, 200)))
        self.assertEqual(1, len(output))
        self.assertEqual(1*10 +
                         (1*1 + 2*100 + 3*200)*11 +
                         (4*1 + 5*100 + 6*200)*12 +
                         (7*1 + 8*100 + 9*200)*13, output[0])
        

class TestInitialWeights(unittest.TestCase):
    
    def setUp(self):
        self.iterable = [1, 2, 3, 4, 5, 6]
        i = 0
        def w():
            i+=1
            return 1
        self.callable = w

    def testInt(self):
        i = InputLayer(3, 2, iweights=3)
        h = HiddenLayer(3, 2, iweights=3)
        expected = [[3, 3, 3], [3, 3, 3]]
        self.assertEqual(expected, i._weights)
        self.assertEqual(expected, h._weights)

    def testIntB(self):
        i = InputLayer(3, 2, iweights=3, bias=1)
        h = HiddenLayer(3, 2, iweights=3, bias=1)
        expected = [[3, 3, 3, 3], [3, 3, 3, 3]]
        self.assertEqual(expected, i._weights)
        self.assertEqual(expected, h._weights)

    def testFloat(self):
        i = InputLayer(3, 2, iweights=2.5)
        h = HiddenLayer(3, 2, iweights=2.5)
        expected = [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]]
        self.assertEqual(expected, i._weights)
        self.assertEqual(expected, h._weights)

    def testFloatB(self):
        i = InputLayer(3, 2, iweights=2.5, bias=1)
        h = HiddenLayer(3, 2, iweights=2.5, bias=1)
        expected = [[2.5, 2.5, 2.5, 2.5], [2.5, 2.5, 2.5, 2.5]]
        self.assertEqual(expected, i._weights)
        self.assertEqual(expected, h._weights)

    def testSeq(self):
        i = InputLayer(3, 2, iweights=[-0.1, 0.1])
        h = HiddenLayer(3, 2, iweights=[-0.1, 0.1])
        for ws in i._weights:
            for w in ws:
                self.assertTrue(w >= -0.1 and w <= 0.1)
        for ws in h._weights:
            for w in ws:
                self.assertTrue(w >= -0.1 and w <= 0.1)

    def testSeqB(self):
        i = InputLayer(3, 2, iweights=[-0.1, 0.1], bias=1)
        h = HiddenLayer(3, 2, iweights=[-0.1, 0.1], bias=1)
        for ws in i._weights:
            for w in ws:
                self.assertTrue(w >= -0.1 and w <= 0.1)
        for ws in h._weights:
            for w in ws:
                self.assertTrue(w >= -0.1 and w <= 0.1)

    def testIterable(self):
        weights = [1, 2, 3, 4, 5, 6]
        i = InputLayer(3, 2, iweights=iter(weights))
        h = HiddenLayer(3, 2, iweights=iter(weights))
        self.assertEqual([[1, 2, 3], [4, 5, 6]], i._weights)
        self.assertEqual([[1, 2, 3], [4, 5, 6]], h._weights)

    def testIterableB(self):
        weights = [1, 2, 3, 4, 5, 6, 7, 8]
        i = InputLayer(3, 2, iweights=iter(weights), bias=1)
        h = HiddenLayer(3, 2, iweights=iter(weights), bias=1)
        self.assertEqual([[1, 2, 3, 4], [5, 6, 7, 8]], i._weights)
        self.assertEqual([[1, 2, 3, 4], [5, 6, 7, 8]], h._weights)

    def testCallable(self):
        r = iter(range(1, 1000))
        i = InputLayer(3, 2, iweights=lambda: next(r))
        r = iter(range(1, 1000))
        h = HiddenLayer(3, 2, iweights=lambda: next(r))
        self.assertEqual([[1, 2, 3], [4, 5, 6]], i._weights)
        self.assertEqual([[1, 2, 3], [4, 5, 6]], h._weights)

    def testCallableB(self):
        r = iter(range(1, 1000))
        i = InputLayer(3, 2, iweights=lambda: next(r), bias=1)
        r = iter(range(1, 1000))
        h = HiddenLayer(3, 2, iweights=lambda: next(r), bias=1)
        self.assertEqual([[1, 2, 3, 4], [5, 6, 7, 8]], i._weights)
        self.assertEqual([[1, 2, 3, 4], [5, 6, 7, 8]], h._weights)

class TestSigmoid(unittest.TestCase):
    def test0(self):
        self.assertTrue(sigmoid(-100) > -1)
        self.assertTrue(sigmoid(100) < 1)


if __name__ == "__main__":
    unittest.main()
