import cv2
import numpy as np
from fitgeotransProj import fitgeotransProj


def py_subpixelFine(P1, P2):
    P1 = np.float32(P1)
    P2 = np.float32(P2)
    # 亚像素级的粗匹配
    length = np.max(P1.shape)
    # 投影变换
    # affmat = cv2.getPerspectiveTransform(P1, P2)
    affmat = fitgeotransProj(P1,P2)
    affmat = affmat.T
    P2pro = np.dot(np.append(P1,np.ones((length,1)),axis=1), affmat.T)
    P2pro = np.append(np.float32([P2pro[:,0]/P2pro[:,2]]).T, np.float32([P2pro[:,1]/P2pro[:,2]]).T,axis=1)
    # 差的平方
    devia_P = (P2-P2pro)**2
    # 按第二维求和开根
    devia_P = np.sqrt(np.sum(devia_P, axis=1))
    max_Devia = np.max(devia_P)
    iteration = 0
    P2fine = P2
    while max_Devia > 0.05 and iteration < 20:
        iteration = iteration+1
        index = sorted(range(len(devia_P)), key=lambda k: devia_P[k])
        ind1 = np.round(1/4*len(index))
        P2fine[index[int(ind1-1):P2fine.shape[0]]] = P2pro[index[int(ind1-1):len(index)]]
        # affmat = cv2.getPerspectiveTransform(P1, P2fine)
        affmat = fitgeotransProj(P1,P2fine)
        affmat = affmat.T
        P2pro = np.dot(np.append(P1,np.ones((length,1)),axis=1), affmat.T)
        P2pro = np.append(np.float32([P2pro[:,0]/P2pro[:,2]]).T, np.float32([P2pro[:,1]/P2pro[:,2]]).T,axis=1)
        # 差的平方
        devia_P = (P2 - P2pro) ** 2
        # 按第二维求和开根
        devia_P = np.sqrt(np.sum(devia_P, axis=1))
        max_Devia = np.max(devia_P)
    return P2fine