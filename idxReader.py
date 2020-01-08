import numpy as np

def readIdxFile(filePath):
    f = open(filePath, 'rb')
    f.read(2)
    dt = int.from_bytes(f.read(1), 'big')
    if dt == 0x8:
        dt = np.ubyte
    elif dt == 0x9:
        dt = np.byte
    elif dt == 0xB:
        dt = np.short
    elif dt == 0xC:
        dt = np.int
    elif dt == 0xD:
        dt = np.single
    elif dt == 0xE:
        dt = np.double
    else:
        raise Exception('Unrecognized data type')
    n = f.read(1)
    n = int.from_bytes(n,'big')
    shape = []
    for i in range(n):
        k = f.read(4)
        shape.append(int.from_bytes(k,'big'))
    arr = np.fromfile(f, dtype=dt)
    arr = arr.reshape(*shape)
    return arr
