import numpy as np
from normalizeCP import normalizeCP

# 得到的结果是仿射矩阵的转置
def fitgeotransProj(uv,xy):
    uv, normMatrix1 = normalizeCP(uv)
    xy, normMatrix2 = normalizeCP(xy)
    minRequiredNonCollinearPairs = 4
    M = xy.shape[0]
    x = xy[:,0]
    y = xy[:,1]
    vec_1 = np.ones(M)
    vec_0 = np.zeros(M)
    u = uv[:,0]
    v = uv[:,1]

    U = np.array(np.append(u,v))

    X1 = np.append(x, vec_0)
    X2 = np.append(y, vec_0)
    X3 = np.append(vec_1, vec_0)
    X4 = np.append(vec_0, x)
    X5 = np.append(vec_0, y)
    X6 = np.append(vec_0,vec_1)
    X7 = np.append(-u*x, -v*x)
    X8 = np.append(-u*y, -v*y)
    X = np.append(np.array([X1]).T, np.array([X2]).T, axis=1)
    X = np.append(X, np.array([X3]).T, axis=1)
    X = np.append(X, np.array([X4]).T, axis=1)
    X = np.append(X, np.array([X5]).T, axis=1)
    X = np.append(X, np.array([X6]).T, axis=1)
    X = np.append(X, np.array([X7]).T, axis=1)
    X = np.append(X, np.array([X8]).T, axis=1)
    
    try:
        # 这个地方应该qr分解！！！！！！！！！！！！
        Tvec = np.dot(np.linalg.pinv(X), U)
    except ValueError as err:
        print(err)
        print('矩阵秩不够')

    Tvec = np.append(Tvec, 1)

    Tinv = np.reshape(Tvec, (3,3))

    Tinv = Tinv.T

    Tinv = np.dot(np.linalg.pinv(normMatrix2), np.dot(Tinv,normMatrix1))

    T = np.linalg.pinv(Tinv)

    T = T / T[2,2]
    return T

