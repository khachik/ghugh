
def readd(iterable):
    data = []
    for line in iterable:
        line = line.strip()
        if not line: return data
        for c in line:
            data.append(c == ' ' and 1 or 0)
    return data

def readi(iterable):
    result = []
    for line in iterable:
        line = line.strip('\r\n')
        if not line:
            continue
        value = line
        result.append((value, readd(iterable)))
    return result

def read(filename):
    with open(filename, 'r') as f:
        return readi(f)

import sys, pprint

if __name__ == "__main__":
    pprint.pprint(read(sys.argv[1]))
