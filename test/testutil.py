import unittest

from ghugh.util import *

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

if __name__ == "__main__":
    unittest.main()
