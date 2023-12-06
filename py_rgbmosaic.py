import numpy as np
from py_graymosaic import py_graymosaic


def py_rgbmosaic(I1, I2, affmat):
    Imosaic = np.append(np.array([py_graymosaic(I1[:, :, 0], I2[:, :, 0], affmat).T]),np.array([py_graymosaic(I1[:, :, 1], I2[:, :, 1], affmat).T]), axis=0)
    Imosaic = np.append(Imosaic, np.array([py_graymosaic(I1[:, :, 2], I2[:, :, 2], affmat).T]), axis=0)
    Imosaic = Imosaic.T
    return Imosaic