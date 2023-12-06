import numpy as np
from curve_tangent import curve_tangent

#                        边
def py_getCorner(curve, curve_start, curve_end, curve_mode, curve_num, BW, sig, Endpoint, C, T_angle, maxlength, rflag):
    corner_num = 0
    cout = np.array([])
    angle = np.array([])
    ang_num = 0


    GaussianDieOff = .0001
    # 1到30的一个数组
    pw = np.array(range(1,31))
    # 3x3
    ssq = sig*sig
    # 满足高斯分布大于阈值的最大的点作为宽度
    width = np.max(np.where(np.exp(-1*(pw*pw)/(2*ssq))>GaussianDieOff))+1
    if width.size==0:
        width = 1
    t = np.array(range(-width, width+1))
    gau = np.exp(-t*t/(2*ssq))
    gau = gau/np.sum(gau)
    # maxlength = 6
    sigmaLs = maxlength

    for i in range(curve_num):
        x = curve[i][:,0]
        y = curve[i][:,1]

        W = width
        # 边缘中，点的个数
        L = len(x)
        if L>W:
            # 计算曲率，xL，yL是n维行向量
            if curve_mode[i]=='loop':
                xL = np.append(x[L-W:L],x)
                xL = np.append(xL,x[0:W])
                yL = np.append(y[L-W:L], y)
                yL = np.append(yL, y[0:W])
            else:
                xL = np.append(np.ones(W)*2*x[0]-x[W:0:-1],x)
                xL = np.append(xL, np.ones(W)*2*x[L-1]-x[L-2:L-W-2:-1])
                yL = np.append(np.ones(W)*2*y[0]-y[W:0:-1],y)
                yL = np.append(yL, np.ones(W) * 2 * y[L - 1] - y[L - 2:L - W - 2:-1])
            # 对xL进行高斯卷积
            xx = np.convolve(xL+1, gau)
            # 提取高2W+L次系数
            xx = xx[W:L+3*W]
            yy = np.convolve(yL+1, gau)
            yy = yy[W:L+3*W]
            xx = np.ceil(xx*10e10)/10e10
            yy = np.ceil(yy*10e10)/10e10
            if i==12:
                temp = np.array([])
                # with open('file.txt', 'r', encoding='utf-8') as f:
                #     for i in f.readlines():
                #         temp = np.append(temp, float(i.strip('\n')))
                #         # print('asdf', yy[73]-yy[71]==1)
                #         # print('72-70', yy[72]-yy[70]==1)
                #         # print(yy[73]-yy[71])
                #         print(i)


            # a2-a1 (a3-a1)/2 an-a(n-1) 一阶微分，n维行向量
            Xu = np.append(np.append(xx[1]-xx[0], (xx[2:L+2*W]-xx[0:L+2*W-2])/2), xx[L+2*W-1]-xx[L+2*W-2])
            Yu = np.append(np.append(yy[1]-yy[0], (yy[2:L+2*W]-yy[0:L+2*W-2])/2), yy[L+2*W-1]-yy[L+2*W-2])


            # a2-a1 a3-a1 an-a(n-1) 二阶微分
            Xuu = np.append(np.append(Xu[1]-Xu[0], (Xu[2:L+2*W]-Xu[0:L+2*W-2])/2), Xu[L+2*W-1]-Xu[L+2*W-2])
            Yuu = np.append(np.append(Yu[1]-Yu[0], (Yu[2:L+2*W]-Yu[0:L+2*W-2])/2), Yu[L+2*W-1]-Yu[L+2*W-2])


            K = np.abs((Xu*Yuu-Xuu*Yu)/(np.power(Xu*Xu+Yu*Yu, 1.5)))
            if i==233:
                print('stop')
            # 向下取整
            K = np.ceil(K*100)/100




            # extremum中存储的是极值点在K中的序号，K中应该是求出的梯度
            extremum = np.array([])
            N = K.shape[0]
            n = 0
            Search = 1

            for j in range(N-1):
                if (K[j+1]-K[j])*Search>0:
                    # n维行向量
                    extremum = np.append(extremum, j)
                    n += 1
                    Search = -Search

            if np.mod(extremum.shape[0], 2)==0:
                extremum = np.append(extremum, N-1)



            # n极值点的个数
            n = extremum.shape[0]
            # flag 表示极值点中大于阈值的点，为1
            flag = np.ones(np.size(extremum))
            lambda_LR = np.array([])
            flag_temp = False
            # 与自适应局部阈值进行比较以去除圆角
            for k in range(1, n, 2):
                if extremum[k-1] == 0:
                    index1 = np.argmin(K[int(extremum[k])::-1])
                else:
                    index1 = np.argmin(K[int(extremum[k]):int(extremum[k-1])-1:-1])
                index2 = np.argmin(K[int(extremum[k]):int(extremum[k+1])+1])

                # index1是在K中第extrk个点到第extrek-1个点中所有点的最小值的序号，曲率最小值点，就是辅助点
                # index2是在K中第extrk个点到第extrek+1个点中所有点的最小值的序号
                ROS = K[int(extremum[k]-index1):int(extremum[k]+index2+1)]
                # ROS的平均值再乘以一个c，以此为阈值
                K_thre = C*np.sum(ROS)/ROS.shape[0]
                K_thre = np.round(K_thre*1e15)/1e15
                if K[int(extremum[k])] < K_thre:
                    flag[k] = 0
                else:
                    if flag_temp:
                        # 极值点，左右最小值点在K中的序号
                        lambda_LR = np.append(lambda_LR, np.array([[extremum[k], index1, index2]]), axis=0)
                    else:
                        lambda_LR = np.array([[extremum[k], index1, index2]])
                        flag_temp = True

            # 获得极大值，因为extremum是极大和极小间隔的
            extremum = extremum[1:n:2]
            flag = flag[1:n:2]
            # 留下的极值点都是曲率值大于阈值的点
            extremum = extremum[np.where(flag == 1)]

            # 检查拐角角度以消除由于边界噪点和琐碎细节而导致的假角  nx2的卷积平滑后的矩阵
            smoothed_curve = np.array([xx, yy])
            smoothed_curve = smoothed_curve.T
            # n表示极值点的个数
            n = extremum.shape[0]
            flag = np.ones(np.size(extremum))
            # 获取极值点所对应的主方向，单位为°
            for j in range(n): 
                # 极值点只有一个的情况下
                # if j == 8 and i ==233:
                #     print('stop')
                #     print(smoothed_curve[int(extremum[j-1]):int(extremum[j+1]+1)])
                if j==0 and j==n-1:
                    ang = curve_tangent(smoothed_curve[0:L+2*W,:], extremum[j])
                elif j==0:
                    # 切片索引
                    ang = curve_tangent(smoothed_curve[0:int(extremum[j+1]+1)], extremum[j])
                elif j==n-1:
                    ang = curve_tangent(smoothed_curve[int(extremum[int(j-1)]):(L+2*W)], extremum[j]-extremum[j-1])
                else:
                    # 输入曲线上前一个极值点到后一个极值点的所有点， 极值点到前一个极值点的距离
                    ang = curve_tangent(smoothed_curve[int(extremum[j-1]):int(extremum[j+1]+1)], extremum[j]-extremum[j-1])
                # T_angle 值为170，即170<ang<190
                if ang>T_angle and ang<(360-T_angle):
                    flag[j] = 0

            # 筛选出轮廓角不与x轴平行的轮廓角对应的序号
            extremum = extremum[np.where(flag!=0)]
            # lambda_LR中存放角度满足要求的点的序号
            lambda_LR = lambda_LR[np.where(flag!=0)]


            # 极值点的每个序号减去一个最大高斯宽度
            extremum = extremum-W
            true_corner = np.array([])
            # 序号小于边缘点的个数，且大于0，对应的点才是真的点
            true_corner = np.where(np.logical_and(extremum>=0,extremum<L))

            # 筛选出真点
            extremum = extremum[true_corner]
            # 找出真点对应的极值和最小值的序号
            lambda_LR = lambda_LR[true_corner]
            # 边缘极值点的个数
            n = extremum.shape[0]
            for j in range(n):

                # cout就是极值点的坐标的集合
                if cout.shape[0]==0:
                    cout = np.array([curve[i][int(extremum[j])]])
                else:
                    # 第i条边， 第extremum[j]个点的坐标填进去
                    cout = np.append(cout, np.array([curve[i][int(extremum[j])]]), axis=0)
                if rflag:
                    # 适应参数算法，此时xcor表示极值点的y坐标，也是距离x轴的距离，ycor是距离y轴的距离
                    xcor = curve[i][int(extremum[j]), 1]
                    ycor = curve[i][int(extremum[j]), 0]
                    # 边缘i的长度
                    retail = len(curve[i])
                    # index1 与极值到边缘最左边点距离小的，包括极值点
                    lengthL = np.min(np.array([lambda_LR[j][1], extremum[j]]))+1
                    # index2 小的 作为左右的长度
                    lengthR = np.min(np.array([lambda_LR[j][2], retail - extremum[j]-1]))+1
                    # 高斯分布 nx1
                    coefL = np.exp(-np.power((np.array(range(int(lengthL)))/lengthL),2)/2)
                    # 每一项高斯除以总和
                    coefL = coefL/np.sum(coefL)
                    coefR = np.exp(-np.power((np.array(range(int(lengthR)-1,-1,-1))/lengthR) ,2)/2)
                    coefR = coefR/np.sum(coefR)
                    # x和y换位置 高斯与对应坐标进行点乘
                    xL = np.sum(coefL*(curve[i][int(extremum[j]+1-lengthL):int(extremum[j]+1),1])) # -1
                    yL = np.sum(coefL*curve[i][int(extremum[j]+1-lengthL):int(extremum[j]+1),0])
                    # 点到最小值点的所有点的y坐标进行高斯加权后之和
                    xR = np.sum(coefR*curve[i][int(extremum[j]):int(extremum[j]+lengthR),1])
                    yR = np.sum(coefR*curve[i][int(extremum[j]):int(extremum[j]+lengthR),0])
                    # 左边点与该极值点链接的直线
                    vL = np.array([xL-xcor, yL-ycor])
                    vR = np.array([xR - xcor, yR - ycor])
                    # 角平分线的方向
                    vm = vL / np.linalg.norm(vL) + vR / np.linalg.norm(vR)


                    # 分配主方向，xy还是反的
                    deltax = vm[0]
                    deltay = vm[1]
                    if np.isnan(np.arctan(deltay/deltax)):
                        orientation1 = 0
                    # 第一象限，都为正
                    elif deltay >= 0 and deltax >= 0:
                        orientation1 = np.arctan(deltay/deltax)
                    # 第四象限    让他们角度都为正  即取0-2pi
                    elif deltay < 0 and deltax >= 0:
                        orientation1 = np.arctan(deltay/deltax)+2*np.pi
                    # 第二，三象限
                    elif (deltay >= 0 and deltax < 0) or (deltay < 0 and deltax < 0 ):
                        orientation1 = np.arctan(deltay/deltax)+np.pi

                    # 主方向所有角度的值放在angle这个数组中
                    if len(angle)==0:
                        angle = np.array([orientation1])
                    else:
                        angle = np.append(angle, orientation1)


                    ang_num = ang_num+1
        # print(i+1,':',angle.shape[0])


    if Endpoint:
        for i in range(curve_num):
            # 第i条边缘的长度
            retail = curve[i].shape[0]
            if retail>0 and curve_mode[i]=='line':
                # 开始点和检测到的角点进行比较
                # cout的极值点坐标减去起始点，就是起始点到极值点的向量
                compare_corner = cout-np.ones((cout.shape[0],1))*curve_start[i]
                # x,y平方
                compare_corner = np.power(compare_corner, 2)
                # 所有极值点的平方和，就是边缘的起始点到极值点的距离的平方
                compare_corner = compare_corner[:,0] + compare_corner[:,1]
                # 两点最小距离大于5
                if np.min(compare_corner)>25:
                    # 添加每条轮廓的起始点
                    cout = np.append(cout, np.array([curve_start[i]]), axis=0)
                    if rflag:
                        # x和y交换
                        xx = curve[i][0,1]
                        yy = curve[i][0,0]
                        # 生成高斯权重
                        coef2 = np.exp(-np.power((np.array(range(np.min([retail,maxlength])-1,-1,-1))/sigmaLs),2)/2)
                        coef2 = coef2 / np.sum(coef2)
                        # 拿coef2和开始的第retail(边的总点数)个点的x和y相乘，并求和，x与y交换顺序
                        # 这个地方加不加1都没啥区别，因为coef2是归一化的，但yL和deltay要一致
                        xL = np.sum(coef2*(curve[i][np.array(range(np.min([retail,maxlength]))),1]))
                        yL = np.sum(coef2*(curve[i][np.array(range(np.min([retail,maxlength]))),0]))
                        deltax = xL-xx
                        deltay = yL-yy
                        if np.isnan(np.arctan(deltay/deltax)):
                            orientation2 = 0
                        elif deltay>=0 and deltax>=0:
                            orientation2 = np.arctan(deltay/deltax)
                        elif deltay<0 and deltax>=0:
                            orientation2 = np.arctan(deltay/deltax)+2*np.pi
                        else:
                            orientation2 = np.arctan(deltay/deltax)+np.pi

                        angle = np.append(angle, orientation2)

                # cout的极值点减去结束点
                compare_corner = cout - np.ones((cout.shape[0], 1)) * curve_end[i]
                # x,y平方
                compare_corner = np.power(compare_corner, 2)
                # 平方和
                compare_corner = compare_corner[:,0] + compare_corner[:,1]
                # 两点最小距离大于5
                if np.min(compare_corner) > 25:
                    cout = np.append(cout, np.array([curve_end[i]]),axis=0)
                    if rflag:
                        end = curve[i].shape[0]-1
                        xx = curve[i][end][1]
                        yy = curve[i][end][0]
                        coef1 = np.exp(-np.power(((np.array(range(retail-np.max([1,retail-maxlength+1])+1))) / sigmaLs), 2)/2)
                        coef1 = coef1 / np.sum(coef1)
                        # 拿coef1和开始的第retail个点的x和y相乘
                        xL = np.sum(coef1 * curve[i][np.array(range(np.max([1, retail-maxlength+1])-1, retail)),1])
                        yL = np.sum(coef1 * curve[i][np.array(range(np.max([1, retail-maxlength+1])-1, retail)),0])
                        deltax = xL - xx
                        deltay = yL - yy
                        if np.isnan(np.arctan(deltay / deltax)):
                            orientation3 = 0
                        elif deltay >= 0 and deltax >= 0:
                            orientation3 = np.arctan(deltay / deltax)
                        elif deltay < 0 and deltax >= 0:
                            orientation3 = np.arctan(deltay / deltax) + 2 * np.pi
                        else:
                            orientation3 = np.arctan(deltay / deltax) + np.pi

                        angle = np.append(angle, orientation3)
    return cout, angle