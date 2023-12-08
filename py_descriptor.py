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
    t0 = time.time()
    for i in range(M):
        for j in range(N):
            # 令所有角度值都为正值 0-2*pi
            if np.isinf(gradient_y[i][j]/gradient_x[i][j]):
                if gradient_y[i][j]>0:
                    angles[i][j] = np.pi*3/2
                else:
                    angles[i][j] = np.pi/2
            elif np.isnan(gradient_y[i][j]/gradient_x[i][j]):
                angles[i][j]=np.nan
            elif gradient_x[i][j] < 0:
                angles[i][j] = np.arctan(gradient_y[i][j]/gradient_x[i][j]) + np.pi
            elif gradient_x[i][j] > 0 and gradient_y[i][j]>=0:
                angles[i][j] = np.arctan(gradient_y[i][j] / gradient_x[i][j])
            elif gradient_x[i][j] > 0 and gradient_y[i][j]<0:
                angles[i][j] = np.arctan(gradient_y[i][j] / gradient_x[i][j]) + 2*np.pi
    t1 = time.time()
    print('循环花费时间：', t1-t0)
    angle1 = np.zeros_like(img)
    ang_x0 = gradient_x==0
    ang_y0 = gradient_y==0
    ang_yp = gradient_y>0
    angle1[np.logical_and(ang_yp,ang_x0)] = np.pi*3/2
    angle1[np.logical_and(~ang_yp,ang_x0)] = np.pi/2
    angle1[np.logical_and(ang_y0,ang_x0)] = np.nan
    ang_xn = gradient_x<0
    angle1[ang_xn] = np.arctan(gradient_y[ang_xn]/gradient_x[ang_xn])+np.pi
    ang_xp = gradient_x > 0
    ang_yp = gradient_y >= 0
    ang_ypxp = np.logical_and(ang_yp, ang_xp)
    angle1[ang_ypxp] = np.arctan(gradient_y[ang_ypxp]/gradient_x[ang_ypxp])
    ang_ynxp = np.logical_and(~ang_yp, ang_xp)
    angle1[ang_ynxp] = np.arctan(gradient_y[ang_ynxp]/gradient_x[ang_ynxp])+2*np.pi
    t2 = time.time()
    print('矩阵运算花费时间：', t2-t1)

    print(np.any(angle1[~np.isnan(angle1)]==angles[~np.isnan(angles)]))

    # angles = readTxt('.\image\\angles.txt', splt='\t', end='\n', start=0)
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
    for p in range(key_num):
        # 梯度平方和再开根
        t = time.time()
        print('first:'+str(t-t0))
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

        # if pp.size == 0:
        #     print(magnitudes)
        #     print(magnitudes.shape)
        #     print('w:',W, 'yp:',yp,'M:', M,'xp:',xp,'N:',N,'p:',p)
        #     print(keypoints.shape)
        #     print((np.max([-W, -yp])+yp).astype(int) , (np.min([W, M-yp-1])+yp).astype(int),(np.max([-W, -xp])+xp).astype(int),(np.min([W, N-xp-1])+xp).astype(int))
        #     #  [0.75 1.   1.   ... 0.   0.   0.  ]]
        #
        #     # 0.0 0.06733539931561396 0.0 0.15288855119425324
        #     print(pp)

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
        tempi=0
        temp = np.array([])
        xv = 0
        t = time.time()
        print('second:'+str(t-t0))
        for kk in range(np.size(dx)):
            print('kk=',kk)
            t = time.time()
            print('third:' + str(t - t0))
            # 这个地方是M-yp-1过来的，所以不应该再加1
            mag = magnitude[int(yp+dy[int(kk%dy.shape[0]),int(kk//dy.shape[0])])][int(xp+dx[int(kk%dx.shape[0]),int(kk//dx.shape[0])])]
            # angles是梯度角，theta是轮廓角
            angle = angles[int(yp+dy[int(kk%dy.shape[0]),int(kk//dy.shape[0])])][int(xp+dx[int(kk%dx.shape[0]),int(kk//dx.shape[0])])]
            # if kk==1435:
            #     print(int(yp+dy[int(kk%dy.shape[0]),int(kk//dy.shape[0])]))
            #     print(int(xp+dx[int(kk%dx.shape[0]),int(kk//dx.shape[0])]))
            #     print('stop')
            # 求出夹角
            angle = np.mod(angle-theta[p], np.pi)
            #三次插值 NBO=8
            nt = NBO*angle/np.pi
            # 向下取整，将连续数据离散化
            binx = np.floor(nx[int(kk%nx.shape[0]),int(kk//nx.shape[0])]-0.5)
            biny = np.floor(ny[int(kk%ny.shape[0]),int(kk//ny.shape[0])]-0.5)
            bint = np.floor(nt)
            # 与n.5的差值
            rbinx = nx[int(kk%nx.shape[0]),int(kk//nx.shape[0])]-(binx+0.5)
            rbiny = ny[int(kk%ny.shape[0]),int(kk//ny.shape[0])]-(biny+0.5)
            # 把nt的小数部分取出来
            rbint = nt-bint
            adf=0

            t = time.time()
            print('4th:' + str(t - t0))

            for dbinx in range(2):
                for dbiny in range(2):
                    for dbint in range(2):
                        # NBP/2=2
                        t = time.time()
                        print('4.5th:' + str(t - t0))
                        if binx+dbinx>=-(NBP/2) and binx+dbinx<(NBP/2) and \
                            biny+dbiny>=-(NBP/2) and biny+dbiny<(NBP/2) and not np.isnan(bint):
                            # （e的（x平方＋y平方））/8*x梯度平方和y梯度平方和*高斯*三个绝对值，获得主方向的权重

                            t = time.time()
                            print('5th:' + str(t - t0))
                            weight = wincoef[int(binx+dbinx+NBP/2)][int(biny+dbiny+NBP/2)]*mag*np.abs(1-dbinx-rbinx)\
                                *np.abs(1-dbiny-rbiny)*np.abs(1-dbint-rbint)

                            # if p==0 and (binx+dbinx+NBP/2)==0 and (biny+dbiny+NBP/2)==0 and np.mod(bint+dbint,NBO)==0:
                            #     ddd = descriptor[int(binx+dbinx+NBP/2)][int(biny+dbiny+NBP/2)]\
                            #     [int(np.mod(bint+dbint,NBO))]+weight
                            #     temp = np.append(temp,ddd)
                            #     tempi = tempi+1
                            #     print(tempi)
                            #     print('kk:',kk)
                            #     if tempi==39:
                            #         print(wincoef[int(binx+dbinx+NBP/2)][int(biny+dbiny+NBP/2)])
                            #         print(binx, biny, bint)
                            #         print(rbinx,rbiny,rbint)


                            t = time.time()
                            print('6th:' + str(t - t0))
                            descriptor[int(binx+dbinx+NBP/2)][int(biny+dbiny+NBP/2)][int(np.mod(bint+dbint,NBO))]=\
                            descriptor[int(binx+dbinx+NBP/2)][int(biny+dbiny+NBP/2)][int(np.mod(bint+dbint,NBO))]+weight

                            t = time.time()
                            print('7th:' + str(t - t0))


            t = time.time()
            print('8th:' + str(t - t0))

        t = time.time()
        print('9th:' + str(t - t0))
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

