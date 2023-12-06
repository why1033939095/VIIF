import numpy as np

def py_atan(dy, dx):
    # 建立nx1的零矩阵
    tanvalue = np.zeros((len(dy),1))
    for i in range(len(dy)):
        if dy[i]==0 and dx[i]==0:
            tanvalue[i] = 90
            continue
        else:
            tanvalue[i] = round(180/np.pi*np.arctan(dy[i]/dx[i]))

    return tanvalue