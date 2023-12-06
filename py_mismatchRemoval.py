import numpy as np
from getEdgeOverlappingRatio import getEdgeOverlappingRatio
import cv2

def py_mismatchRemoval(p1, p2, edge1, edge2, maxErr):
    '''
    :param p1: IR点
    :param p2: VI点
    :param edge1:
    :param edge2:
    :param maxErr:maxRMSE = 4*ceil(size(I2,1)/300)
    :return:
    '''
    iteration = 1
    zoomflag = 1
    p1 = p1.T
    p2 = p2.T
    if len(p1) == 3:
        correctIndex = np.array([0,1,2])
        regis2sub = p2
        return correctIndex
    if len(p2) == 2:
        correctIndex = np.array([0,1])
        return correctIndex
    correctIndex = np.array([])
    #像素点个数/80
    minArea1 = edge1.shape[0]*edge1.shape[1]/80
    minArea2 = edge2.shape[0]*edge2.shape[1]/80
    length = p1.shape[0]
    eor = np.array([])

    # RANSAC算法
    for n in range(min(length*(length-1)*(length-2), 500)):
        # 随机从0到length-1取三个整数
        ijk = np.random.randint(0,length,3)
        ir = ijk[0]
        jr = ijk[1]
        kr = ijk[2]


        # 随机选出三个点
        pp1 = np.array([p1[ir,0:2], p1[jr, 0:2], p1[kr, 0:2]])
        pp2 = np.array([p2[ir,0:2], p2[jr, 0:2], p2[kr, 0:2]])
        # 两点求差
        A1 = p1[ir]-p1[jr]
        B1 = p1[ir]-p1[kr]
        A2 = p2[ir]-p2[jr]
        B2 = p2[ir]-p2[kr]

        if np.abs(A1[0]*B1[1]-A1[1]*B1[0])<minArea1 or np.abs(A2[0]*B2[1]-A2[1]*B2[0])<minArea2:
            continue

        ransacErr = getEdgeOverlappingRatio(pp1, pp2, p1[:,0:2], p2[:, 0:2], ir, jr, kr, maxErr)
        if len(eor)==0:
            eor = np.array([np.append(np.array([ir, jr, kr]), ransacErr)])
        else:
            eor = np.append(eor,np.array([np.append(np.array([ir,jr,kr]),ransacErr)]),axis=0)
    # 小于三个点
    if eor.shape[0]<3:
        raise ValueError('匹配不足， 获得不足三个匹配')

    ind = np.argmax(eor[:,4])
    base1 = p1[eor[int(ind), 0:3].astype(int),0:2]
    base2 = p2[eor[int(ind), 0:3].astype(int),0:2]
    affmat0 = cv2.getAffineTransform(np.float32(base1),np.float32(base2))
    affmat0 = np.append(affmat0, np.array([[0,0,1]]), axis=0)
    correctIndex = eor[ind, 0:3]
    for i in range(length):
        if i == eor[ind][0] or i == eor[ind][1] or i == eor[ind][2]:
            continue
        pp1_aff = np.dot(np.append(np.array([p1[i,0:2]]),np.array([1])),affmat0.T)
        # p2-p1的仿射矩阵的绝对值
        pp2_to_pp1aff = np.abs(p2[i,0:2]-pp1_aff[0:2])
        if (pp2_to_pp1aff[0]<maxErr/1.5 and pp2_to_pp1aff[1]<1.5*maxErr) or (pp2_to_pp1aff[0]<1.5*maxErr and pp2_to_pp1aff[1]<maxErr/1.5):
            correctIndex = np.append(correctIndex, np.array([i]))

    return correctIndex