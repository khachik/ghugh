import sys

from ghugh import nn
from data import read

def testXOR():
    net = nn.Net(2, 3, 1, bias=True)

    data = (
            ((1, 1), (0,)),
            ((1, 0), (1,)),
            ((0, 1), (1,)),
            ((0, 0), (0,))
           )

    for i in range(50000):
        for input, v in data:
            print(input, list(net.feed(input)))
            net.learn(input, v, N=0.1)
        print("Iteration #%d passed" % i)
    for input, _ in data:
        print(input, list(net.feed(input)))

    sys.exit(0)

def testSymmetry():
    IN = 2
    ON = 2
    HN = 2
    net1 = nn.Net(IN, HN, ON)
    net2 = nn.Net(IN, HN, ON)
    o2 = (1, 1)
    o1 = (0, 1)
    data = [
            [o1 + (0,)*(IN - len(o1)),
             (1,0) + (0,)*(ON - 2)],
            [o2 + (0,)*(IN - len(o2)),
             (0,1) + (0,)*(ON - 2)]
           ]
    for i in range(1):
        for (i1, v1), (i2, v2) in zip(data, data[::-1]):
            print("Training net1")
            net1.learn(i1, v1, N=0.5)
            print("Training net2")
            net2.learn(i2, v2, N=0.5)
            print("Activating F")
            print("F", v1, net1.feed(i1)[:3])
            print("Activating B")
            print("B", v2, net2.feed(i2)[:3])
        print("Iteration #%d passed" % i)

def testDATA():
    net = nn.Net(81, 45, 10, bias=True)
    print(net)
    data = read.read(sys.argv[1])

    d = data[:]
    for i in range(500):
        for v, input in d:
            value = [0] * 10
            value[int(v)] = 1
            net.learn(input, value, N=0.5)
        print("Iteration #%d passed" % i)
    for v, input in d:
        value = [0] * 10
        value[int(v)] = 1
        print(v, [round(o) for o in net.feed(input)])

    ONE="""*********
**** ****
**** ****
**** ****
**** ****
**** ****
**** ****
**     **
*********""".split("\n")
    TWO="""*********
**     **
** *** **
***** ***
**** ****
*** *****
** *** **
**     **
*********""".split("\n")

    one = read.readd(ONE) 
    two = read.readd(TWO)

    print("1", [round(o) for o in net.feed(one)])
    print("2", [round(o) for o in net.feed(two)])

if __name__ == "__main__":
    #testSymmetry()
    #testDATA()
    testXOR()
