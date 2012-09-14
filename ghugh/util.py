import itertools, functools, collections

def compose(*functions, unpack=False):
    if unpack:
        l = lambda x, y: y(*x)
    else:
        l = lambda x, y: y(x)
    def f(*args, **kwargs):
        return functools.reduce(l, itertools.islice(functions, 1, None), \
                         functions[0](*args, **kwargs))
    return f

class _Cursor(collections.MutableSequence):
    def __init__(self, target, index):
        self.count = len(target)
        self.index = index

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return self.target[i][self.index]

    def __setitem__(self, i, value):
        target[i][self.index] = value

    def __delitem__(self, i):
        del target[i][self.index]

class _Transposed(collections.Sequence):
    def __init__(self, target):
        self.columns = tuple(Cursor(target, i) for i in range(len(target)))

    def __len__(self):
        return len(self.columns)

    def __getitem__(self, i):
        return self.columns[i]
