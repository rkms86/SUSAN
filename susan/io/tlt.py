__all__ = ['read']

import numpy as _np

def read(filename):
    return _np.loadtxt(filename)