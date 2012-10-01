import itertools, functools, collections

def compose(*functions, unpack=False):
    """Function composition as:
    composed(x) -> functions[n](...(functions[1](functions[0](x))))
    if unpack is True, then the returned value from each call is unpacked
    befored passing to the next function."""

    if unpack:
        l = lambda x, y: y(*x)
    else:
        l = lambda x, y: y(x)
    def f(*args, **kwargs):
        return functools.reduce(l, itertools.islice(functions, 1, None), \
                         functions[0](*args, **kwargs))
    return f

class _Cursor(collections.MutableSequence):
    """Mutable orthogonal view (e.g. a column in 2-dim matrix represented
    as a list of rows):
    m = [[1, 2, 3],
         [4, 5, 6]]
    c = _Cursor(m, 2) # c = [3,
                      #      6]
    c[0] = 3

    """

    def __init__(self, target, index):
        self.count = len(target)
        self.index = index
        if index >= len(target[0]):
            raise IndexError("%d is out of [0, %d)" % (index, self.count))
        self.target = target

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return self.target[i][self.index]

    def __setitem__(self, i, value):
        self.target[i][self.index] = value

    def __delitem__(self, i):
        raise NotImplementedError

    def insert(self, i, value):
        raise NotImplementedError

class _Transposed(collections.Sequence):
    """Transposed mutable view of 2-dim matrix.
    m = [[1, 2, 3],
         [4, 5, 6]]
    mT = [[1, 4],
          [2, 5],
          [3, 6]]
    """

    def __init__(self, target):
        self.columns = tuple(_Cursor(target, i)
                             for i in range(len(target[0])))

    def __len__(self):
        return len(self.columns)

    def __getitem__(self, i):
        return self.columns[i]

def transposed(matrix):
    """2-dim matrix transposition."""

    return _Transposed(matrix)
