import numpy as np

def normalizeCP(pts):
    N = pts.shape[0]
    cent = np.mean(pts, axis=0)
    ptsNorm = pts - cent
    sumOfPD = np.sum(np.power(ptsNorm[:,0],2)+np.power(ptsNorm[:,1],2))

    if sumOfPD > 0:
        scaleFactor = np.sqrt(2*N)/np.sqrt(sumOfPD)
    else:
        scaleFactor = 1

    ptsNorm = ptsNorm*scaleFactor
    normMatrixInv = np.array([[1/scaleFactor,0,0],[0,1/scaleFactor,0],[cent[0],cent[1],1]])
    return ptsNorm, normMatrixInv