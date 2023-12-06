import numpy as np

def py_match(des1, des2, distRatio):
    # des1：特征向量个数*4*4*8 特征子
    # des2: 缩放比例*特征向量长度*特征向量个数
    # 3*放缩次数
    zoomvote = np.zeros((3, des2.shape[0]))
    # 特征向量个数*放缩次数
    match1 = np.zeros((des1.shape[0], des2.shape[0]))
    # 特征向量个数*缩放次数
    match2 = np.zeros((des2.shape[2], des2.shape[0]))
    # 在第二张图中找第一张图匹配上的的描述子
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            des2t = des2[j,:,:]
            dotprods = np.dot(des1[i,:], des2t)
            # 对描述子之积去反余弦，然后排序
            vals1 = np.sort(np.arccos(dotprods), axis=0)
            indx1 = sorted(range(len(dotprods)), key=lambda k:np.arccos(dotprods[k]))
            # 比2pi大就行
            zoomvote[0][j] = 10
            # 第二个值*0.97大于第一个值，记录并匹配，图1匹配上的点，记录下其对应点在乘法之后的结果的坐标
            if vals1[0] < distRatio * vals1[1]:
                # 对应2的不同比例，存入每种比例中最小反余弦值
                zoomvote[0][j] = vals1[0]
                # 1中的每个描述子乘以2中不同的缩放比例图像，符合条件的第i个描述子，2的比例，反余弦最小值的序号存入match1对应位置
                match1[i][j] = indx1[0]+1

        # 描述子之积反余弦最小的缩放比例，
        mintheta = np.min(zoomvote[0])
        # 最小值所在缩放比例的坐标
        ind = np.argmin(zoomvote[0])
        if mintheta != 10:
            # 最小值不等于10，在第二维度对应的缩放比例序号上加一，表示在不同缩放比例满足要求最小角度的特征子个数（已经更新了的个数）
            zoomvote[1][ind] = zoomvote[1][ind]+1

    # 将1,2都进行转置操作，让2中的每个特征向量和1直接矩阵乘法，执行相同的操作，match2中存储对应坐标
    des1t = des1.T
    for j in range(des2.shape[0]):
        des2zoom = des2[j,:,:].T
        for i in range(des2zoom.shape[0]):
            # 拿图2的特征向量乘以图1的每一个特征向量 
            dotprods = np.dot(des2zoom[i], des1t)
            vals1 = np.sort(np.arccos(dotprods))
            indx1 = sorted(range(len(dotprods)), key=lambda k:np.arccos(dotprods[k]))

            if vals1[0] < distRatio * vals1[1]:
                match2[i][j] = indx1[0]
        # shape[0]是特征点的个数
    # 双边匹配
    # 图1的描述子个数*1*2的行向量
    zoomvote[2] = des1.shape[0] * np.ones((1, des2.shape[0]))
    temp = np.array([])
    for j in range(des2.shape[0]):
        for i in range(des1.shape[0]):
            # 有坐标的点才有可能是匹配上的最小值
            if match1[i][j]>0:
                # match2中最小值的序号和match1中的最小值不一样, 那么就对应位置减一，剩下的就是每张图不同缩放系数匹配上的特征描述子。i就是向量的坐标
                # 即match1和match2中都需要相等
                # 从match1中取出的坐标在match2当中的索引后的值是不是match1的索引，一样的话那就匹配上了
                if match2[int(match1[i][j]-1)][j] != i:
                    match1[i][j] = 0
                    zoomvote[2][j] = zoomvote[2][j]-1                   # 不为零说明第j次缩放的图的第i个描述子中match有不为零的值
            else:
                zoomvote[2][j] = zoomvote[2][j] - 1               # 匹配上的描述子个数

    # 匹配点数个数，匹配描述子个数最多的缩放比例
    maxvote = np.max(zoomvote[2])
    zoom = np.argmax(zoomvote[2])
    # 获取匹配上的坐标
    correctIndex1 = np.array([])
    correctIndex2 = np.array([])

    for i in range(des1.shape[0]):
        # zoom缩放的第i个描述子,匹配到的最多的缩放比例
        if match1[i][zoom] > 0:
            if len(correctIndex1)==0:
                correctIndex1 = np.array([i])
                # 1*（2的所有描述子）的反余弦的最大值的序号（对应于图2）
                correctIndex2 = np.array([match1[i][zoom]-1])
            else:
                correctIndex1 = np.append(correctIndex1, i)
                correctIndex2 = np.append(correctIndex2, match1[i][zoom]-1)

    #    行向量，图1的第几个特征向量    行向量，图2匹配上的特征向量     一个数缩放倍数
    return correctIndex1, correctIndex2, zoom