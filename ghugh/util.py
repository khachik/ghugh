import itertools, functools

def compose(*functions, unpack=False):
    if unpack:
        l = lambda x, y: y(*x)
    else:
        l = lambda x, y: y(x)
    def f(*args, **kwargs):
        return functools.reduce(l, itertools.islice(functions, 1, None), \
                         functions[0](*args, **kwargs))
    return f
