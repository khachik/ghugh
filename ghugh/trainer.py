from abc import abstractmethod

class Algo(object):
    """Training algorithm interface."""

    @abstractmethod
    def train(self, dataset, LR, M):
        raise NotImplementedError

def supervised(algo, dataset,
               learningRate, momentum,
               epoches, E=0.001):
    """Supervised training on the given dataset (a sequence of
    2-element tuples). Returns a tuple of (converged, error)."""

    for i in range(epoches):
        e = algo.train(dataset, learningRate, momentum)
        if e <= E:
            return True, e
    return False, e
