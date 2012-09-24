import sys

from ghugh import ffann
from ghugh import trainer
from ghugh import backprop
from data import read


def testXOR():
    input = ffann.InputLayer(2, 2, iweights=1, bias=True)
    hidden = ffann.HiddenLayer(2, 1, iweights=1, bias=True)
    output = ffann.OutputLayer(1)
    net = ffann.Net(input, hidden, output)

    data = (
            ((1, 1), (0,)),
            ((1, 0), (1,)),
            ((0, 1), (1,)),
            ((0, 0), (0,))
           )

    print("Initial +++++++++++++++")
    for input, v in data:
        print(v, list(net.feed(input)))
    print()

    print("Traning...")
    algo = backprop.Backpropagation(net, False)
    converged, error = trainer.supervised(algo, data, 1.0, 0.0, 2000)
    print("Converged:", converged,)
    print("Error:", error)
    for input, v in data:
        print(v, list(net.feed(input)))

    sys.exit(0)

def testDATA():
    input = ffann.InputLayer(81, 45, iweights=0, bias=True)
    hidden = ffann.HiddenLayer(45, 10, iweights=0, bias=True)
    output = ffann.OutputLayer(10)
    net = ffann.Net(input, hidden, output)

    d = read.read(sys.argv[1])
    data = []
    for v, input in d:
        value = [0] * 10
        value[int(v)] = 1
        data.append([input, value])

    print("Initial +++++++++++++++")
    for input, v in data:
        print(v, list(net.feed(input)))
    print()

    print("Traning...")
    algo = backprop.Backpropagation(net, False)
    converged, error = trainer.supervised(algo, data, 0.1, 0.0, 500)
    print("Converged:", converged,)
    print("Error:", error)
    for input, v in data:
        print(v, list(net.feed(input)))


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
    testDATA()
    #testXOR()
