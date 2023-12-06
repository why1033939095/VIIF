from normalizeCP import normalizeCP
import numpy as np

def fitgeotransAffi(uv, xy):
    uv, normMatrix1 = normalizeCP(uv)
    xy, normMatrix2 = normalizeCP(xy)
