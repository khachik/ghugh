import math, unittest, collections

from ghugh.nn import *

class Weighted(collections.Sequence):
    def __init__(self, inputs, weights):
        self._weights = weights
        self._inputs=  inputs

    def weights(self, index):
        return self._weights[index]

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, index):
        return self._inputs[index]


class TestOutputLayer(unittest.TestCase):
    def setUp(self):
        self.o = OutputLayer(5,
                             function=lambda x: 1000*x + 100000, 
                             dfunction=lambda x: 1000)
        self.inputs = [1, 2, 3]
        self.weights = [[4, 5, 6], [7, 8, 9], [10, 11, 12],
                   [13, 14, 15], [16, 17, 18]]
        self.data = Weighted(self.inputs, self.weights)

    def test_activate(self):
        response = list(self.o.activate(self.data))
        self.assertEqual(5, len(response))
        self.assertEqual((4*1 + 5*2 + 6*3)*1000 + 100000, response[0])
        self.assertEqual((7*1 + 8*2 + 9*3)*1000 + 100000, response[1])
        self.assertEqual((10*1 + 11*2 + 12*3)*1000 + 100000, response[2])
        self.assertEqual((13*1 + 14*2 + 15*3)*1000 + 100000, response[3])
        self.assertEqual((16*1 + 17*2 + 18*3)*1000 + 100000, response[4])

        
    def test_fix(self):
        response = list(self.o.activate(self.data))
        deltas = self.o.fix(response)
        self.assertEqual(len(self.o), len(deltas))
        self.assertTrue(all(d == 0 for d in deltas))
        deltas = self.o.fix([10, 20, 30, 40, 50])
        self.assertEqual(len(self.o), len(deltas))
        self.assertEqual(1000*(10 - ((4*1 + 5*2 + 6*3)*1000 + 100000)),
                         deltas[0])
        self.assertEqual(1000*(20 - ((7*1 + 8*2 + 9*3)*1000 + 100000)),
                         deltas[1])
        self.assertEqual(1000*(30 - ((10*1 + 11*2 + 12*3)*1000 + 100000)),
                         deltas[2])
        self.assertEqual(1000*(40 - ((13*1 + 14*2 + 15*3)*1000 + 100000)),
                         deltas[3])
        self.assertEqual(1000*(50 - ((16*1 + 17*2 + 18*3)*1000 + 100000)),
                         deltas[4])


class TestInputLayer(unittest.TestCase):

    def setUp(self):
        self.i = InputLayer(3, 4)
        self.i._weights = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    def test_activate(self):
        response = list(self.i.activate([10, 20, 30]))
        self.assertEqual(3, len(response))
        self.assertEqual(10, response[0])
        self.assertEqual(20, response[1])
        self.assertEqual(30, response[2])
            

    def test_fix(self):
        self.i.activate([112, 223, 334])
        self.i.activate([10, 20, 30])
        self.i.fix([100, 200, 300, 400], N=0.4)

        w = list(self.i.weights(0))
        self.assertEqual(3, len(w))
        self.assertEqual(401, w[0])
        self.assertEqual(802, w[1])
        self.assertEqual(1203, w[2])

        w = list(self.i.weights(1))
        self.assertEqual(3, len(w))
        self.assertEqual(804, w[0])
        self.assertEqual(1605, w[1])
        self.assertEqual(2406, w[2])

        w = list(self.i.weights(2))
        self.assertEqual(3, len(w))
        self.assertEqual(1207, w[0])
        self.assertEqual(2408, w[1])
        self.assertEqual(3609, w[2])

        w = list(self.i.weights(3))
        self.assertEqual(3, len(w))
        self.assertEqual(1610, w[0])
        self.assertEqual(3211, w[1])
        self.assertEqual(4812, w[2])

class TestHiddenLayer(unittest.TestCase):

    def setUp(self):
        self.h = HiddenLayer(3, 2,
                             function = lambda x: 11*x + 12,
                             dfunction = lambda x: 11)
        self.h._weights = [[1, 2, 3], [4, 5, 6]]
        self.weighted = Weighted([10, 20, 30, 40],
                                 [[100, 200, 300, 400],
                                  [500, 600, 700, 800],
                                  [900, 1000, 1100, 1200]]
                                 )

    def test_activate(self):
        response = list(self.h.activate(self.weighted))
        self.assertEqual(3, len(response))
        self.assertEqual(11*(10*100 + 20*200 + 30*300 + 40*400) + 12,
                         response[0])
        self.assertEqual(11*(10*500 + 20*600 + 30*700 + 40*800) + 12,
                         response[1])
        self.assertEqual(11*(10*900 + 20*1000 + 30*1100 + 40*1200) + 12,
                         response[2])

    def test_fix(self):
        self.h.activate(self.weighted)
        deltas = list(self.h.fix([0, 0]))
        self.assertEqual(3, len(deltas))
        self.assertEqual(0, deltas[0])
        self.assertEqual(0, deltas[1])
        self.assertEqual(0, deltas[2])

        self.assertEqual([1, 2, 3], list(self.h.weights(0)))
        self.assertEqual([4, 5, 6], list(self.h.weights(1)))

        deltas = list(self.h.fix([1000, 2000], 0.5))

        w = list(self.h.weights(0))
        self.assertEqual(3, len(w))
        self.assertEqual(1 + 500*(11*(10*100 + 20*200 + 30*300 + 40*400)
                                  + 12),
                         w[0])
        self.assertEqual(2 + 500*(11*(10*500 + 20*600 + 30*700 + 40*800)
                                  + 12),
                         w[1])
        self.assertEqual(3 + 500*(11*(10*900 + 20*1000 + 30*1100 + 40*1200)
                                  + 12),
                         w[2])

        w = list(self.h.weights(1))
        self.assertEqual(3, len(w))
        self.assertEqual(4 + 1000*(11*(10*100 + 20*200 + 30*300 + 40*400)
                                   + 12),
                         w[0])
        self.assertEqual(5 + 1000*(11*(10*500 + 20*600 + 30*700 + 40*800)
                                   + 12),
                         w[1])
        self.assertEqual(6 + 1000*(11*(10*900 + 20*1000 + 30*1100 + 40*1200)
                                   + 12),
                         w[2])

class TestSigmoid(unittest.TestCase):
    def test0(self):
        self.assertTrue(sigmoid(-100) >= -1)
        self.assertTrue(sigmoid(100) <= 1)

if __name__ == "__main__":
    unittest.main()
