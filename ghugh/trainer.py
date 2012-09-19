
class Algo(object):
    def train(dataset):
        raise NotImplementedError

def supervised(net, algo, dataset, epoches, E=0.001):
    for i in range(epoches):
        e = algo.train(dateset)
        if e <= E:
            return True, e
    return False, e
