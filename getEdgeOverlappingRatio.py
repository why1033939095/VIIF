import cv2
import numpy as np


def getEdgeOverlappingRatio(pp1, pp2, p1, p2, a, b, c, maxErr):
    pp1 = np.float32(pp1+1)
    pp2 = np.float32(pp2+1)
    affmat = cv2.getAffineTransform(pp1, pp2)
    affmatT = np.append(affmat.T, np.array([[0,0,1]]).T, axis=1)
    p1_aff = np.dot(np.append(p1+1,np.ones((p1.shape[0], 1)),axis=1),affmatT)
    # rows, cols = p1_aff.shape
    # p1_aff = np.dot(p1_aff, affmat.T)
    # p2-p1_aff的平方
    p2_to_p1aff = np.power((p2+1-p1_aff[:,0:2]),2)

    RMSE = np.zeros((1,4))
    # 平方和除以P2长宽大的值，再开根
    RMSE[0][0] = np.sqrt(np.sum(p2_to_p1aff)/np.max(p2.shape))
    # 两矩阵之差，xy的平方和
    pixelDistance = p2_to_p1aff[:,0]+p2_to_p1aff[:,1]
    # 将值替换成2maxErr^2
    pixelDistance[a] = 2 * np.power(maxErr, 2)
    pixelDistance[b] = 2 * np.power(maxErr, 2)
    pixelDistance[c] = 2 * np.power(maxErr, 2)
    # 距离小于maxErr^2的点
    u = np.where(pixelDistance<np.power(maxErr,2))[0]
    # 最小距离
    minerr = np.min(pixelDistance)
    # 最小距离的点数
    ind0 = np.argmin(pixelDistance)
    # [平均值， 距离满足要求点的个数， 最小像素距离， 最小像素距离坐标]
    RMSE[0,1:4] = np.array([len(u), minerr, ind0])

    return RMSE