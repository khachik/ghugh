import unittest

from ghugh.util import compose, _Cursor, _Transposed, transposed

class TestCompose(unittest.TestCase):
        
    def testPacked(self):
        functions=[lambda x: x+1, lambda x: x+10,
                   lambda x: x+100, lambda x: x+1000]
        c = compose(*functions)
        self.assertEqual(1111, c(0))

    def testUnpacked(self):
        functions = []
        functions=[lambda x,y: (x+1, y+1),
                   lambda x,y: (x+10, y+100),
                   lambda x,y: (x+100, y+10000),
                   lambda x,y: (x+1000, y+1000000)]
        c = compose(*functions, unpack=True)
        self.assertEqual((1111, 1010101), c(0, 0))


class TestCursor(unittest.TestCase):
    def setUp(self):
        self.data = [[1, 2, 3, 4],
                     [5, 6, 7, 8]]

    def testLen(self):
        c = _Cursor(self.data, 0)
        self.assertEqual(2, len(c))
        c = _Cursor(self.data, 3)
        self.assertEqual(2, len(c))

    def testOutOfBounds(self):
        c = _Cursor(self.data, 4)
        self.assertRaises(IndexError, c.__getitem__, 0)
        c = _Cursor(self.data, 3)
        self.assertRaises(IndexError, c.__getitem__, 2)

    def testValues(self):
        c = _Cursor(self.data, 0)
        self.assertEqual(1, c[0])
        self.assertEqual(5, c[1])

        c = _Cursor(self.data, 3)
        self.assertEqual(4, c[0])
        self.assertEqual(8, c[1])

    def testSetItem(self):
        c = _Cursor(self.data, 0)
        self.assertRaises(IndexError, c.__setitem__, 2, 11)
        self.assertEqual(5, c[1])
        c[1] = 11
        self.assertEqual(11, c[1])
        self.assertEqual(11, self.data[1][0])


class TestTransposed(unittest.TestCase):
    def setUp(self):
        self.matrix = [[1, 2, 3, 4],
                       [5, 6, 7, 8]]
        self.tmatrix = _Transposed(self.matrix)

    def testLen(self):
        self.assertEqual(len(self.matrix[0]), len(self.tmatrix))
        self.assertEqual(len(self.matrix), len(self.tmatrix[0]))

    def testValues(self):
        self.assertEqual(self.matrix[1][3], self.tmatrix[3][1])

    def testTT(self):
        tt = transposed(self.tmatrix)
        for i,r in enumerate(self.matrix):
            for j,c in enumerate(r):
                self.assertEqual(c, tt[i][j])


if __name__ == "__main__":
    unittest.main()
