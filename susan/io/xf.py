__all__ = ['read']

import numpy as _np

def read(filename, parse = True):
    data = _np.loadtxt(filename)
    if parse:
        A11, A12, A21, A22, DX, DY = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        data = _np.stack([_np.stack([A11,A12,DX], axis = -1), _np.stack([A21,A22,DY], axis = -1)], axis = 1)
    return data