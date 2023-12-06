import numpy as np
import matplotlib.pyplot as plt
import cv2


def py_showMatch(I1, I2, loc1, loc2, correctPos, tname):
    cols = I1.shape[1]
    if I1.shape[0]<I2.shape[0]:
        I1_p = np.append(I1,np.zeros((I2.shape[0]-I1.shape[0], I1.shape[1], I1.shape[2])), axis=0).astype(int)
        im3 = np.append(I1_p,I2, axis=1)
    elif I1.shape[0]>I2.shape[0]:
        I2_p = np.append(I2, np.zeros((I1.shape[0]-I2.shape[0], I2.shape[1], I2.shape[2])) , axis=0)
        im3 = np.append(I1, I2_p, axis=1)
    else:
        im3 = np.append(I1, I2, axis=0)

    if len(correctPos):
        for i in range(correctPos.shape[0]):
            # cv2.arrowedLine()
            cv2.line(im3, loc1[i], [loc2[i][0]+cols,loc2[i][1]], (255, 255, 0), 1, 0, 0, 0.05)
            cv2.circle(im3, loc1[i], 3, (255, 0, 0))
            cv2.circle(im3, loc2[i]+np.array([cols,0]), 3, (0, 255, 0))

        for i in range(loc1.shape[0]):
            cv2.line(im3, loc1[i], loc2[i]+np.array([cols, 0]),(255,0,0),1,0,0,0.05)
            cv2.circle(im3, loc1[i],3,(255,0,0))
            cv2.circle(im3, loc2[i]+np.array(cols,0), 3,(255,255,0) )
    else:
        for i in range(loc1.shape[0]):
            cv2.line(im3, np.round(loc1[i]).astype(int), np.round(loc2[i]+np.array([cols, 0])).astype(int),(0,255,0),1)
            cv2.circle(im3, np.round(loc1[i]).astype(int), 3, (255,0,0))
            cv2.circle(im3, np.round(loc2[i]).astype(int), 3, (255,255,0) )

    plt.imshow(im3)
    plt.title(tname)
    plt.show()