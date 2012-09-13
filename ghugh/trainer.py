import nn2


def odeltas(layer, expected):
    e = iter(expected)
    return [layer.dfunction(o)*(next(e) - o) for o in layer]

def deltas(layer, odeltas):
    deltas = [0.0] * len(layer)
    deltaws = []
    for oi, od in enumerate(odeltas):
        weights = layer.weights(oi)
        deltaw = []
        for (i, (o, w)) in enumerate(zip(layer, weights)):
            deltas[i] += od * w
            deltaw.append(od*o)
        deltaws.append(deltaw)
    return deltas, deltaws
 

def train(net, data, epoches, R, M):
    for i in range(epoches):
        dws = [[0.0] * 
        for input, expected in data:

