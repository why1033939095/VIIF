import numpy as np

def py_extractCurve(BW, Gap_size):
    (L,W) = np.array(BW).shape
    # 即在图片周围加了一圈像素点，便于用3x3的窗口对每个像素点进行遍历处理


    BW1 = np.zeros((L+2*Gap_size, W+2*Gap_size))
    # BW_edges
    BW_edge = np.zeros_like(BW)
    # 把edges放edges1中间
    BW1[Gap_size:Gap_size+L, Gap_size:Gap_size+W] = BW
    # 找出边缘点的横纵坐标
    # # (r,c) = np.where(BW1==255)
    BW1 = BW1.T
    (c,r) = np.where(BW1==255)
    BW1 = BW1.T

    # 记录边缘个数
    cur_num = 0
    curve = []
    while r.shape[0]>0:
        # point 是点的坐标
        point = np.array([r[0], c[0]])
        # point = np.array([122,2])
        # cur是一条边缘的所有点的列表
        cur = np.array([point])
        # 把检测出来的点先清零，为下一步找最近点做准备
        BW1[point[0], point[1]]=0
        # 在point周围八个点中像素值为255的点的坐标
        # # (I, J) = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1, point[1]-Gap_size:point[1]+Gap_size+1]==255)
        BW1 = BW1.T
        (J, I) = np.where(BW1[point[1]-Gap_size:point[1]+Gap_size+1, point[0]-Gap_size:point[0]+Gap_size+1]==255)
        BW1 = BW1.T

        while I.shape[0]>0:
            # dist是周围像素到中心点的距离的平方
            dist = np.square(I - Gap_size)+np.square(J - Gap_size)
            # axis=0是以列向量输入，返回每列的最小值的坐标
            index = np.argmin(dist, axis=0)
            # point更新为最近的一个检测出来的点
            point = point+np.array([I[index],J[index]])-Gap_size
            # 在cur中添加这个点的坐标
            cur = np.append(cur, [point], axis=0)
            BW1[point[0], point[1]] = 0
            # I是第一个维度的参数，所以对应的是y坐标，J是x坐标
            # # (I, J) = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1, point[1]-Gap_size:point[1]+Gap_size+1]==255)
            BW1 = BW1.T
            (J, I) = np.where(
                BW1[point[1] - Gap_size:point[1] + Gap_size + 1, point[0] - Gap_size:point[0] + Gap_size + 1] == 255)
            BW1 = BW1.T


        # 处理了这个点就把这个点的值归零，即从次距离近的点开始再遍历
        point = np.array([r[0], c[0]])
        BW1[point[0], point[1]] = 0
        # 在point周围八个点中像素值为255的点的坐标
        # # (I, J) = np.where(BW1[point[0] - Gap_size:point[0] + Gap_size + 1, point[1] - Gap_size :point[1] + Gap_size + 1] == 255)
        BW1 = BW1.T
        (J, I) = np.where(
            BW1[point[1] - Gap_size:point[1] + Gap_size + 1, point[0] - Gap_size:point[0] + Gap_size + 1] == 255)
        BW1 = BW1.T
        while I.shape[0] > 0:
            # dist是周围像素到中心点的距离的平方
            dist = np.square(I - Gap_size) + np.square(J - Gap_size)
            # axis=0是以列向量输入，返回每列的最小值的坐标，dist就是一个行向量
            index = np.argmin(dist, axis=0)
            # point更新为最近的一个点
            point = point + np.array([I[index], J[index]]) - Gap_size
            # 往第一个点之前添加点
            cur = np.append(np.array([point]),cur, axis=0)
            BW1[point[0], point[1]] = 0
            # # (I, J) = np.where(BW1[point[0] - Gap_size:point[0] + Gap_size + 1, point[1] - Gap_size
            # #                  :point[1] + Gap_size + 1] == 255)
            BW1 = BW1.T
            (J, I) = np.where(BW1[point[1] - Gap_size:point[1] + Gap_size + 1, point[0] - Gap_size:point[0] + Gap_size + 1] == 255)
            BW1 = BW1.T


        # cur中是点的集合，边缘点的个数大于图像长+宽/60，那么将其算作一条轮廓
        if cur.shape[0]>(BW.shape[0]+BW.shape[1])/60:
            cur_num+=1
            # 因为每一条边缘上的点的个数不一样，所以不能用数组表示，因此curve是list，list中的元素是nx2的数组
            curve.append(cur-Gap_size)

        # 重新更新r和c
        # # (r,c) = np.where(BW1 == 255)
        BW1 = BW1.T
        (c, r) = np.where(BW1 == 255)
        BW1 = BW1.T

    curve_start = np.array([])
    curve_end = np.array([])
    curve_mode = np.array([])
    for i in range(cur_num):
        # 获取的是点在边缘的点集里的索引
        if i==0:
            curve_start = np.array([curve[i][0]])
            curve_end = np.array([curve[i][curve[i].shape[0] - 1]])
        else:
            # 边缘的第一个点
            curve_start = np.append(curve_start, [curve[i][0]], axis=0)
            # 边缘的最后一个点
            curve_end = np.append(curve_end, [curve[i][curve[i].shape[0] - 1]], axis=0)


        # 边缘的第一个点和最后一个点的距离小于等于2，判定为环，轮廓闭合，否则判定为一条边
        if np.square(curve_start[i][0]-curve_end[i][0])+np.square(curve_start[i][1]-curve_end[i][1])<=4:
            curve_mode = np.append(curve_mode, ['loop'])
        else:
            curve_mode = np.append(curve_mode, ['line'])

        # 还原边缘图像
        BW_edge[curve[i][:,0], curve[i][:,1]] = 255

    return curve, curve_start, curve_end, curve_mode, cur_num, BW_edge