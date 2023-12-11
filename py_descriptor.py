import cv2
import numpy as np
import copy
import time
from readTxt import readTxt

# sift算法
def py_descriptor(img, keypoints):
    NBP = 4
    NBO = 8
    # 特征点的个数
    key_num = keypoints.shape[1]
    # 4x4x8行，key_num列
    descriptors = np.zeros((NBP*NBP*NBO, key_num))

    M, N = img.shape
    du_filter = np.array([[-1, 0, 1]])
    # 进行转置
    dv_filter = du_filter.T
    # 对角化矩阵
    duv_filter = np.diag(np.array([-1,0,1]))
    # 进行一次镜像
    dvu_filter = np.flip(duv_filter,axis=1)
    # 利用自定义的滤波器对原图进行滤波
    # 第二个参数是目标图像深度（数据类型）直接设置成-1，img要转化成float，而且左右两列不参与运算，所以要加两列0
    img = np.append(np.zeros((1,N)).astype(float), img, axis=0)
    img = np.append(img, np.zeros((1,N)).astype(float), axis=0)
    img = np.append(np.zeros((M+2,1)).astype(float), img, axis=1)
    img = np.append(img, np.zeros((M+2,1)).astype(float), axis=1)
    # img = np.round(img)
    # 第二个参数是期望深度，-1表示输出类型和输入相同
    gradient_u = cv2.filter2D(img, -1, du_filter)
    gradient_u = np.round(gradient_u * np.power(10, 4)) / np.power(10, 4)
    gradient_v = cv2.filter2D(img, -1, dv_filter)
    gradient_v = np.round(gradient_v * np.power(10, 4)) / np.power(10, 4)
    gradient_uv = cv2.filter2D(img, -1, duv_filter)
    gradient_uv = np.round(gradient_uv * np.power(10, 4)) / np.power(10, 4)
    gradient_vu = cv2.filter2D(img, -1, dvu_filter)
    gradient_vu = np.round(gradient_vu * np.power(10, 4)) / np.power(10, 4)
    # 根据得到的结果进行计算
    gradient_x = 1.414*gradient_u + gradient_uv - gradient_vu
    gradient_y = 1.414*gradient_v + gradient_uv + gradient_vu
    img = img[1:M+1, 1:N+1]
    magnitudes = np.sqrt(np.power(gradient_x, 2)+np.power(gradient_y, 2))
    magnitudes = magnitudes[1:M+1,1:N+1]
    gradient_x = gradient_x[1:M+1,1:N+1]
    gradient_y = gradient_y[1:M+1,1:N+1]

    angles = np.zeros(img.shape)
    # 计算梯度方向

    angles = np.zeros_like(img)
    ang_x0 = gradient_x==0
    ang_y0 = gradient_y==0
    ang_yp = gradient_y>0
    angles[np.logical_and(ang_yp,ang_x0)] = np.pi*3/2
    angles[np.logical_and(~ang_yp,ang_x0)] = np.pi/2
    angles[np.logical_and(ang_y0,ang_x0)] = np.nan
    ang_xn = gradient_x<0
    angles[ang_xn] = np.arctan(gradient_y[ang_xn]/gradient_x[ang_xn])+np.pi
    ang_xp = gradient_x > 0
    ang_yp = gradient_y >= 0
    ang_ypxp = np.logical_and(ang_yp, ang_xp)
    angles[ang_ypxp] = np.arctan(gradient_y[ang_ypxp]/gradient_x[ang_ypxp])
    ang_ynxp = np.logical_and(~ang_yp, ang_xp)
    angles[ang_ynxp] = np.arctan(gradient_y[ang_ynxp]/gradient_x[ang_ynxp])+2*np.pi


    # keypoints是4行的矩阵，1.x坐标 2.y坐标 3.比例 4.角度
    x = keypoints[0]
    y = keypoints[1]
    s = keypoints[2]
    theta = keypoints[3]
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    a = np.linspace(-NBP/2, NBP/2, NBP+1)
    b = np.linspace(-NBP/2, NBP/2, NBP+1)
    xx, yy = np.meshgrid(a,b)
    wincoef = np.exp(-(np.power(xx,2)+np.power(yy,2))/np.power(NBP,2)*2)

    t0 = time.time()
    print('begin:'+str(t0))
    t_xunhuan = 0
    t_buxunhuan = 0
    for p in range(key_num):
        if p==1000:
            print('1000个特征点计算循环耗时：',t_xunhuan)
            print('1000个特征点计算非循环耗时：', t_buxunhuan)
        # 梯度平方和再开根
        magnitude = copy.deepcopy(magnitudes)
        sp = s[p]   # scale=36
        xp = x[p]
        yp = y[p]
        sinth0 = sintheta[p]
        costh0 = costheta[p]
        W = sp
        # ss=18
        ss = W/2

        # 4*4*8
        descriptor = np.zeros((NBP, NBP, NBO))

        # M 是图像高度,截出一个矩形区域，yp比36大，yp-36到36+yp+1，xp也是相同的道理
        pp = magnitudes[(np.max([-W, -yp])+yp).astype(int) : (np.min([W, M-yp-1])+yp+1).astype(int), \
             (np.max([-W, -xp])+xp).astype(int):(np.min([W, N-xp-1])+xp+1).astype(int)]


        pp = pp/np.max(pp)
        # 按列升序排序，全部拉通来排
        xx = np.sort(np.reshape(pp,(pp.size,)))

        # 把xx分成五段，每段进行四舍五入
        xind1 = np.round((1-1/5)*xx.shape[0])-1
        xind2 = np.round((1-2/5)*xx.shape[0])-1
        xind3 = np.round((1-3/5)*xx.shape[0])-1
        xind4 = np.round((1-4/5)*xx.shape[0])-1
        # 看pp落在xx（从小到大排列）中的什么位置0，0.25，0.5，0.75，1
        pp = ((pp>=xx[int(xind1)])+np.logical_and(pp<xx[int(xind1)], pp>=xx[int(xind2)])*0.75+np.logical_and(pp<xx[int(xind2)],\
                pp>=xx[int(xind3)])*0.5+np.logical_and(pp<xx[int(xind3)], pp>=xx[int(xind4)])*0.25)

        magnitude[int(np.max([-W,-yp])+yp):int(np.min([W, M-yp-1])+yp+1),int(np.max([-W,-xp])+xp):int(np.min([W, N-xp-1])+xp+1)] = pp

        # 建立一个网格
        a = np.linspace(int(np.max([-W, -xp])), int(np.min([W, N-xp-1])), int(np.min([W, N-xp-1])-np.max([-W, -xp])+1))
        b = np.linspace(int(np.max([-W, -yp])), int(np.min([W, M-yp-1])), int(np.min([W, M-yp-1])-np.max([-W, -yp])+1))
        dx, dy = np.meshgrid(a,b)

        # 网格x正弦加余弦，ss为比例/2，即18
        nx = (costh0*dx+sinth0*dy)/ss
        ny = (-sinth0*dx+costh0*dy)/ss



        ddy = np.ravel(yp+dy.T).astype(int)
        ddx = np.ravel(xp+dx.T).astype(int)
        mag1 = magnitude[ddy, ddx]
        angle1 = angles[ddy, ddx]
        angle1 = np.mod(angle1-theta[p], np.pi)
        nt1 = NBO*angle1/np.pi

        nx1 = np.ravel(nx.T)
        ny1 = np.ravel(ny.T)
        binx1 = np.floor(nx1 - 0.5)
        biny1 = np.floor(ny1 - 0.5)
        bint1 = np.floor(nt1)

        rbinx1 = nx1 - binx1 - 0.5
        rbiny1 = ny1 - biny1 - 0.5
        rbint1 = nt1 - bint1


        for kk in range(np.size(dx)):

            mag = mag1[kk]
            angle = angle1[kk]
            nt = nt1[kk]

            binx = binx1[kk]
            biny = biny1[kk]
            bint = bint1[kk]

            rbinx = rbinx1[kk]
            rbiny = rbiny1[kk]
            rbint = rbint1[kk]


            if not np.isnan(bint):
                dbinx = np.array([0, 0, 1, 1])
                dbiny = np.array([0, 1, 0, 1])
                bdbinx = dbinx + binx + NBP / 2
                bdbiny = dbiny + biny + NBP / 2
                flag_x = np.logical_and(bdbinx >= 0, bdbinx < NBP)
                flag_y = np.logical_and(bdbiny >= 0, bdbiny < NBP)
                flag_xy = np.logical_and(flag_x, flag_y)
                weight = wincoef[bdbinx[flag_xy].astype(int), bdbiny[flag_xy].astype(int)] * mag * np.abs( \
                    1 - dbinx[flag_xy] - rbinx) * np.abs(1 - dbiny[flag_xy] - rbiny) * np.abs(1 - rbint)

                descriptor[
                    bdbinx[flag_xy].astype(int), bdbiny[flag_xy].astype(int), np.mod(bint, NBO).astype(int)] += weight
                weight = wincoef[bdbinx[flag_xy].astype(int), bdbiny[flag_xy].astype(int)] * mag * np.abs( \
                    1 - dbinx[flag_xy] - rbinx) * np.abs(1 - dbiny[flag_xy] - rbiny) * np.abs(rbint)
                descriptor[bdbinx[flag_xy].astype(int), bdbiny[flag_xy].astype(int), np.mod(bint + 1, NBO).astype(
                    int)] += weight




        descriptor = descriptor.T
        temp = np.array([])
        for ti in range(NBO):
            temp = np.append(temp,np.reshape(descriptor[ti], NBP*NBP))
        descriptor = temp
        # descriptor = np.reshape(descriptor,(1,NBP*NBP*NBO))
        # 向量归一化
        print(p,':', np.linalg.norm(descriptor))
        descriptor = descriptor/np.linalg.norm(descriptor)
        # 二维以上的数组就要加[]
        descriptors[:, p] = descriptor.T
        t = time.time()
        print('0th:' + str(t - t0))
    return descriptors