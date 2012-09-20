
class Algo(object):
    def train(self, dataset, LR, M):
        raise NotImplementedError

def supervised(algo, dataset,
               learningRate, momentum,
               epoches, E=0.001):
    for i in range(epoches):
        e = algo.train(dataset, learningRate, momentum)
        if e <= E:
            return True, e
    return False, e
