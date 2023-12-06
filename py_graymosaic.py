import numpy as np
import math
from scipy import interpolate
from readTxt import readTxt


def py_graymosaic(I1,I2,affmat):
    r1,c1 = I1.shape
    r2,c2 = I2.shape

    # 呈左上右下进行排列图片
    Imosaic = np.zeros((r2+2*np.max((r1,c1)),c2+2*np.max((r1,c1))))

    affinemat = affmat.T
    # 找出不是数字的部分，即每个点的x和y坐标
    u,v = np.where(~ np.isnan(Imosaic.T))
    # 每个点的减去max(图1长宽)
    v = v-np.max((r1,c1))+1
    u = u-np.max((r1,c1))+1
    
    # x坐标，y坐标，1
    utvt = np.dot(np.append(np.append(np.array([u]).T,np.array([v]).T,axis=1), np.ones((v.shape[0],1)),axis=1),np.linalg.pinv(affinemat))
    ut = utvt[:,0]/utvt[:,2]
    vt = utvt[:,1]/utvt[:,2]
    utu = np.reshape(ut,(c2+2*np.max((r1,c1)), r2+2*np.max((r1,c1)))).T
    vtv = np.reshape(vt,(c2+2*np.max((r1,c1)), r2+2*np.max((r1,c1)))).T

    # 二维插值
    y = np.arange(I1.shape[0])+1
    x = np.arange(I1.shape[1])+1
    # xx, yy = np.meshgrid(x,y)
    # utu = readTxt(r'.\image\utu.txt', '\t')
    # vtv = readTxt(r'.\image\vtv.txt', '\t')
    f = interpolate.interp2d(x, y, I1.astype(float))
    Iterp = np.full(utu.shape, np.nan)
    for i in range(utu.shape[0]):
        for j in range(utu.shape[1]):
            if utu[i][j]-1 >= 0 and vtv[i][j]-1 >= 0 and utu[i][j] - 1 <= 575 \
                    and vtv[i][j] -1 <= 767:
                Iterp[i][j] = f(utu[i][j],vtv[i][j])

    # Iterp = f(utu, vtv)
    un, vn = np.where(~np.isnan(Iterp.T))
    vmin1 = np.min(vn)
    vmax1 = np.max(vn)
    umin1 = np.min(un)
    umax1 = np.max(un)
    Imosaic = np.full(Imosaic.shape,np.nan)
    Imosaic[np.max((r1,c1)):np.max((r1,c1))+r2,np.max((r1,c1)):np.max((r1,c1))+c2] = I2
    for i in range(Imosaic.shape[0]):
        for j in range(Imosaic.shape[1]):
            if not math.isnan(Iterp[i][j]) and not math.isnan(Imosaic[i][j]):
                Imosaic[i][j] = (Imosaic[i][j]+Iterp[i][j])/2
            elif not math.isnan(Iterp[i][j]) and math.isnan(Imosaic[i][j]):
                Imosaic[i][j]=Iterp[i][j]
    validuv = np.array([[np.min((vmin1+1,np.max((r1,c1)))), np.min((umin1+1,np.max((r1, c1))))],[np.max((vmax1+1,np.max((r1,c1))+r2)), np.max((umax1+1, np.max((r1,c1))+c2))]])
    Imosaic = Imosaic[validuv[0][0]-1:validuv[1][0], validuv[0][1]-1:validuv[1][1]].astype(np.uint8)
    return Imosaic