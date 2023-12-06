import numpy as np
from py_cornerDetection import py_cornerDetection
from py_descriptor import py_descriptor
import cv2
from py_mismatchRemoval import py_mismatchRemoval
from py_atan import py_atan
from py_match import py_match
from readTxt import readTxt
import matplotlib.pyplot as plt
from py_resizeImage import imresize
from PIL import Image

# input:            I1, I2, 20,      maxRMSE, 0,         1,          0,           6,  1,        I2gray
def py_registration(I1, I2, maxtheta, maxErr, iteration, zoomascend, zoomdescend, Lc, showflag, I2ori):
    print('\n【1】Image registrating...\n\tplease wait for a few seconds...\n')

    # cor_IR, orientation_IR, IRedge =  \
    BW1 = np.array(plt.imread(r'.\image\BW.png'))
    cor1, orientation1, I1edge = py_cornerDetection(I1, [], [], [], [], 0, 1, 1, Lc, iteration==0, BW1)
    BW2 = np.array(plt.imread(r'.\image\BW1.png'))
    cor2, orientation2, I2edge = py_cornerDetection(I2, [], [], [], [], [], 1, 1, Lc, iteration==0, BW2)
    # coriv = [[cor_ir[:,1], cor_ir[:,0]] , [0,0] , [cor_vi[:,1], cor_vi[:,0]]]
    # 重新将x和y交换回来
    cor12 = np.append(np.array([cor1[:,1]]).T, np.array([cor1[:,0]]).T, axis=1)
    cor12 = np.append(cor12, np.array([[0,0]]), axis=0)
    temp = np.append(np.array([cor2[:,1]]).T, np.array([cor2[:,0]]).T, axis=1)
    # [红外光图,[0,0],可见光图]
    cor12 = np.append(cor12, temp, axis=0)
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    I2ori = I2ori.astype(float)
    # 显示每个角的方向
    if iteration==0 and showflag:
        # 各个角度求出正余弦，轮廓角+2辅助角
        xu1 = 4*np.cos(orientation1)
        yv1 = 4*np.sin(orientation1)
        xu2 = 4*np.cos(orientation2)
        yv2 = 4*np.sin(orientation2)
        # 归一化
        image1 = I1/255
        image2 = I2/255


    # 画图
    scale12 = 36
    # 调换回来了
    # 在这一列的第几个，cols是y轴
    cols1 = cor1[:,0]
    # 在这一行中的第几个，row是x轴
    rows1 = cor1[:,1]
    # n行一列的极值点。乘以比例36
    s1 = scale12 * np.ones((cols1.shape[0],1))
    if iteration>0:     # 没有必要计算方向
        print(cols1.shape[0])
        # 特征点的个数
        o1 = np.zeros(cols1.shape[0])
    else:
        o1 = orientation1

    # 第一行是x 第二行是y 第三行是比例，第四行是角度
    key1 = np.append(np.array([rows1]), np.array([cols1]),axis=0)
    key1 = np.append(key1,s1.T, axis=0)
    key1 = np.append(key1, np.array([o1]), axis=0)

    # 描述子是4*4*8行，n（就是特征点的个数）列的矩阵
    des1 = py_descriptor(I1, key1)
    # 在多尺度的情况下，从图二中提取特征点，交换行和列（本来就是反的）(没反)
    # des1 = readTxt('.\image\des1.txt', splt='   ', end='\n', start=1)
    cols2 = cor2[:,0]
    rows2 = cor2[:,1]
    if iteration > 0:
        o2 = np.zeros(np.size(cols2))
    else:
        o2 = orientation2

    # s_vi = scaleiv * np.ones(np.size(cols_vi))
    # key_vi = np.arrary([rows_vi.T, cols_vi.T, s_vi.T, o_vi.T])
    s2 = scale12 * np.ones((np.size(cols2),1))
    key2 = np.append(np.array([rows2]), np.array([cols2]), axis=0)
    key2 = np.append(key2, s2.T, axis=0)
    key2 = np.append(key2, np.array([o2]), axis=0)
    # 缩放变换参数
    zoomstepup = 1
    zoomstepdown = 0.5
    des2 = np.zeros((zoomascend+zoomdescend+1, 128,np.max(cor2.shape)))
    des2[0,:,:] = py_descriptor(I2, key2)
    # des2[0,:,:] = readTxt('.\image\des2_0.txt', splt='   ', end='\n', start=1)
    level = 0
    scale = I2ori.shape[0]/I2.shape[0]
    if iteration == 0:
        for i in range(zoomascend):
            level = level+1
            key2zoom = key2
            # 以zoomsetup为步长，不断以(1+zoomsetup*i)/scale的比例进行缩放，直接在dsize形参中写(256,512)，则得到的其实是（512,256）的结果
            # I2zoom = cv2.resize(I2ori, (int((1+zoomstepup*(i+1))/scale*I2ori.shape[1]), int((1+zoomstepup*(i+1))/scale*I2ori.shape[0])))
            I2zoom = imresize(I2ori,(1+zoomstepup*(i+1)/scale))
            # I2zoom = readTxt('.\image\I2zoom.txt',splt='\t', end='\n')
            key2zoom[0:2] = np.floor((1+zoomstepup*(i+1))*key2zoom[0:2]+1)
            # python中是第一个参数代表第三个维度
            des2[level,:,:] = py_descriptor(I2zoom, key2zoom)
            # des2[level,:,:] = readTxt('.\image\des2_1.txt', splt='   ', end='\n', start=1)

        for i in range(zoomdescend):
            level = level+1
            key2zoom = key2
            if len(I2ori)==0:
                print('VVIori矩阵为零矩阵')
            # I2zoom = cv2.resize(I2ori, (int((1-zoomstepdown*(i+1))/scale*I2ori.shape[1]), int((1-zoomstepdown*(i+1))/scale*I2ori.shape[0])))
            I2zoom = imresize(I2ori, (1-zoomstepdown*(i+1))/scale)
            key2zoom[0:2] = np.floor((1-zoomstepdown*i)*key2zoom[0:2])
            des2[level,:,:] = py_descriptor(I2zoom,key2zoom)

    # 关键点匹配，通过BBF方法粗略匹配, 1的描述子，2的描述子，zoom就是第几个缩放比例
    matchIndex1, matchIndex2, zoom = py_match(des1.T, des2, 0.97 )
    print('匹配完毕')
    zoomscale = (zoom==0)*1 + (0<zoom and zoom<=zoomascend)*(1+zoom*zoomstepup) + (zoom>zoomascend)*(1-(zoom-zoomascend)*zoomstepdown)
    
    # [x坐标，y坐标， 轮廓角]
    regis_points111 = np.append(np.array([cor1[matchIndex1.astype(int), 1]]).T,np.array([cor1[matchIndex1.astype(int), 0]]).T,axis=1)
    regis_points111 = np.append(regis_points111,np.array([o1[matchIndex1.astype(int)]]).T, axis=1)
    
    regis_points222 = np.append(np.array([cor2[matchIndex2.astype(int), 1]]).T, np.array([cor2[matchIndex2.astype(int), 0]]).T, axis=1)
    regis_points222 = np.append(regis_points222, np.array([o2[matchIndex2.astype(int)]]).T, axis=1)
    # 少于两个点的情况
    if regis_points111.shape[0] < 2:
        raise ValueError('匹配点不够，匹配失败')

    if showflag:
        pass

    if iteration == 0:
        # 图2的轮廓角-图1的轮廓角，从弧度值转化成角度值
        delta0 = np.mod(np.round(180/np.pi*(regis_points222[:,2]-regis_points111[:,2])), 360)

        # 以5为间隔
        dd = 5
        # python360包含在355-360这个区间里 matlab360单独放在一个柱里，下同
        d_delta = range(0,366, dd)
        # 获得的是在每个区间的元素数量，左闭右开
        n_delta,_ = np.histogram(delta0, d_delta)
        # 倒序排列得到序号，元素个数从大到小
        nindex = sorted(range(len(n_delta)), key=lambda k:n_delta[k], reverse=True)
        # 得到角度差众数的序号（角度）
        n0 = nindex[0]
        # 序号的平方，序号，1，序号-1的平方，序号-1，1，序号+1的平方，序号+1，1。右乘nmat，等于众数，除众数之外最多的数的个数，最少的数的个数
        nmat = np.dot(np.linalg.pinv(np.mat([[int(np.power((n0+1),2)), int(n0+1), 1],[int(np.power(n0,2)), int(n0), 1],\
                                             [int(np.power(n0+2,2)), int(n0+2), 1]])),np.mat([n_delta[int(n0)], n_delta[int(n0-1+360/dd*(n0==0))], n_delta[int(n0+1-(n0==(360/dd-1)))]]).T)
        # -b/2a
        Modetheta_discrete = -nmat[1]/2/nmat[0]
        #抛物线插值,modetheta是旋转角]
         
        Modetheta = Modetheta_discrete*dd
    else:
        Modetheta = 0

    # 连线等长平行，那么则认为匹配是正确的
    # x坐标+1-宽度/2，y坐标+1-长度/2 乘 旋转矩阵， +1是为了和matlab程序对应
    trans222 = np.dot(np.append(np.array([regis_points222[:,0]-I2.shape[1]/2+1]).T, np.array([regis_points222[:,1]-I2.shape[0]/2+1]).T, axis=1),\
                      np.array([[np.cos(Modetheta*np.pi/180)[0,0],-np.sin(Modetheta*np.pi/180)[0,0]],\
                                [np.sin(Modetheta*np.pi/180)[0,0], np.cos(Modetheta*np.pi/180)[0,0]]]))
    # 恢复矩阵
    trans222 = np.append(np.array([trans222[:,0]+I2.shape[1]/2-1]).T, np.array([trans222[:,1]+I2.shape[0]/2-1]).T, axis=1)

    # 求出一个角度值（180），由于是反正切函数，所以取值范围在（-90~90） I1和旋转后的I1排一行，旋转前后同一点的连线的反正切值
    phi_uv = py_atan(zoomscale*trans222[:,1]-regis_points111[:,1], zoomscale*trans222[:,0]-regis_points111[:,0]+I1.shape[1])

    if showflag:
        pass

    dd = 5
    # 真的需要改，不然下一步筛选后个数就有区别了
    d_phi = range(-90, 96, dd)
    n_phi,_ = np.histogram(phi_uv,d_phi)
    # 倒序，从大到小
    nindex = sorted(range(len(n_phi)), key=lambda k:n_phi[k], reverse=True)
    n0 = nindex[0]
    # 找n_phi 中多于最大量的0.2倍的值的坐标
    interval_index = np.where(n_phi<(n_phi[n0]*0.2))[0]
    # 这些坐标再减去最大值坐标
    interval_index = np.array(interval_index) - n0
    # 找出坐标小于零的点，那么他们就在这个最大值坐标的左边
    left_phi = np.where(interval_index<0)[0]
    right_phi = np.where(interval_index>0)[0]
    # 最大值左右两个点对应的区间值
    maxtheta1 = -dd*interval_index[left_phi[len(left_phi)-1]]
    maxtheta2 = dd*interval_index[right_phi[0]]
    # 第一个矩阵是纯数组运算，不涉及到矩阵    序号的平方，序号，1，序号-1的平方，序号-1，1，序号+1的平方，序号+1，1。右乘nmat，等于众数，除众数之外最多的数的个数，最少的数的个数
    nmat = np.dot(np.linalg.pinv(np.array([[int(np.power(n0+1,2)),n0+1,1],[np.power(n0+180/dd*(n0==0),2), n0+180/dd*(n0==0), 1],\
                                         [np.power(n0+2-(n0==(180/dd-1)),2),n0+2-(n0==(180/dd-1)),1]])),\
                                          np.array([[n_phi[int(n0)]],[n_phi[int(n0-1+180/dd*(n0==0))]],[n_phi[int(n0+1-(n0==(180/dd-1)))]]]))

    # b/2a-16
    ModePhi_discrete = -nmat[1]/2/nmat[0]-90/dd
    ModePhi = ModePhi_discrete*dd
    delta1 = ModePhi-maxtheta1
    delta2 = ModePhi+maxtheta2
    valid0 = np.where(np.logical_and((phi_uv>=delta1),(phi_uv<=delta2)))[0]


    # 图1和旋转后并缩放的图2中对应点距离，切片一定要写成[x,y]，而不能写成[x][y]
    Dist = np.sqrt(np.power(zoomscale*trans222[valid0, 1]-regis_points111[valid0, 1],2)+\
                    np.power(zoomscale*trans222[valid0, 0]+I1.shape[1]-regis_points111[valid0, 0], 2))
    # 求平均值
    meandist = np.mean(Dist)
    # 距离大于0.5倍平均值小于1.5*缩放比例倍平均值
    valid1 = np.where(np.logical_and((Dist >= 0.5*meandist),(Dist<=1.5*zoomscale*meandist)))[0]

    regis_points11 = regis_points111[valid0[valid1]].T
    regis_points22 = regis_points222[valid0[valid1]].T

    if showflag:
        pass

    correctindex = py_mismatchRemoval(regis_points11, regis_points22, I1, I2, maxErr)
    regis_points11 = regis_points11.T
    regis_points22 = regis_points22.T
    regis_points1 = regis_points11[correctindex.astype(int), 0:2]
    regis_points2 = regis_points22[correctindex.astype(int), 0:2]

    if showflag:
        pass
    return regis_points1, regis_points2, cor12

if __name__ == '__main__':
    py_registration()