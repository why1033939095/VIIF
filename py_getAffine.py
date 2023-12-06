import numpy as np
import cv2
from fitgeotransProj import fitgeotransProj

def py_getAffine(I1, I2, p1, p2):
    p1 = np.float32(p1-1)
    p2 = np.float32(p2-1)
    # I1 = np.append(np.zeros((1,I1.shape[1])), I1, axis=0)
    # I1 = np.append(np.zeros((I1.shape[0],1)), I1, axis=1)
    if p1.shape[0] == 3:
        # 仿射变换
        affmatT = cv2.getAffineTransform(p1, p2)
        Iaffine = cv2.warpAffine(I1, affmatT, (I1.shape[1], I1.shape[0]))
    elif p1.shape[0] >= 4:
        # 透视变换
        # affmat = cv2.getPerspectiveTransform(p1,p2)
        affmatT = fitgeotransProj(p1,p2)
        affmat = affmatT.T
        Iaffine = cv2.warpPerspective(I1, affmat, (I1.shape[1], I1.shape[0]))
    elif p1.shape[0] == 2:
        # 相似变换 https://blog.csdn.net/Roaddd/article/details/112365634
        sca = np.linalg.norm(p1[0]-p1[1])/np.linalg.norm(p2[0]-p2[1])
        ang = np.arccos(np.dot((p1[0]-p1[1]),(p2[0]-p2[1]))/(np.linalg.norm(p1[0]-p1[1])*np.linalg.norm(p2[0]-p2[1])))
        affmatT = cv2.getRotationMatrix2D(center=(I1.shape[0]/2,I1.shape[1]/2),\
                              angle = ang,\
                              scale = sca)
        Iaffine = cv2.warpAffine(I1, affmatT, (I1.shape[1], I1.shape[0]))

    else:
        raise ValueError('Transformation Failed! No sufficient Matches!!')

    # Iaffine = Iaffine[1:Iaffine.shape[0], 1:Iaffine.shape[1]]
    return Iaffine, affmat