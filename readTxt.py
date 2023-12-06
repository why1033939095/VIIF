import numpy as np
def readTxt(filename, splt, end='\n', num=None, start=0):
    '''
    :param filename: 文件路径
    :param splt: 分隔符
    :param end: 结尾符
    :param num: 保留浮点数位数
    :return: 数组
    '''
    with open(filename,'r',encoding='utf-8') as f:
        dataf = f.readlines()
        flag = 1
        dataa = np.array([])
        for datal in dataf:
            atemp = np.array([])
            datal = datal.strip(end).split(splt)
            for i in datal[start:len(datal)]:
                atemp = np.append(atemp, float(i))
            if flag:
                dataa = np.array([atemp])
                flag = 0
            else:
                dataa = np.append(dataa,np.array([atemp]), axis=0)
        if num is not None:
            dataa = np.round(dataa*np.power(10,num))/np.np.power(10,num)
    return dataa