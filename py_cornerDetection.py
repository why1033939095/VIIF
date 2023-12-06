import cv2

import numpy as np
from py_extractCurve import py_extractCurve
from py_getCorner import py_getCorner
import matplotlib.pyplot as plt

#                      IR, [], [],    [], [], 0,  1,       1,         Lc=62w3q ,       true
def py_cornerDetection(I, C, T_angle, sig, H, L, Endpoint, Gap_size, maxlength, rflag, BW):
    if len(C)==0:
        C = 1.5
    if len(T_angle)==0:
        T_angle = 170
    if len(sig)==0:
        sig = 3
    if len(H)==0:
        # H = 0.2
        H = 256
        # H = 0.5805*255
    if type(L)==type([]) and len(L)==0:
        L=0

    # canny边缘检测，传给canny的必须要是uint8
    BW = cv2.Canny(np.uint8(I), 0, 126)
    # plt.imshow(BW)
    # plt.show()
    # BW = np.array(plt.imread(r'.\image\BW.png'))*255
    # BW = BW*255


    # k=0
    # x = np.loadtxt("file.txt",delimiter=' ')
    # adf = x.shape
    # for i in edges:
    #     for j in i:
    #         if j==255:
    #             k+=1
    # k
    # plt.imshow(edges,cmap = 'gray')
    # plt.show()



    curve, curve_start, curve_end, curve_mode, curve_num, edge_map = py_extractCurve(BW, Gap_size)
    # angle就是n个主方向，n个首点方向，n个尾点方向
    cout, angle = py_getCorner(curve, curve_start, curve_end, curve_mode, curve_num, BW, sig, Endpoint, C, T_angle, maxlength, rflag)
    return cout, angle, edge_map