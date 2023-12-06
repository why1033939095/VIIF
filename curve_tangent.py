import numpy as np


def curve_tangent(cur, center):
    direction = np.array([])
    for i in range(2):
        # 极值点到第一个点
        if i==0:
            curve = cur[int(center)::-1]
        else:
            # 极值点到最后一个点
            curve = cur[int(center):cur.shape[0],:]

        L = curve.shape[0]

        if L>3:
            # 判断首尾是否相同
            if np.sum(cur[0]!=cur[L-1])!=0:
                # M是点个数的一半
                M = np.ceil(L/2)-1
                x1 = curve[0][0]
                y1 = curve[0][1]
                x2 = curve[int(M)][0]
                y2 = curve[int(M)][1]
                x3 = curve[L-1][0]
                y3  = curve[L-1][1]
            else:
                # 一条边分成三段，x和y是二次卷积的结果
                M1 = np.ceil(L/3)-1
                M2 = np.ceil(2*L/3)-1
                x1 = curve[0][0]
                y1 = curve[0][1]
                x2 = curve[M1][0]
                y2 = curve[M1][1]
                x3 = curve[M2][0]
                y3 = curve[M2][1]

            # 12和13是否平行
            if np.abs((x1-x2)*(y1-y3)-(x1-x3)*(y1-y2))<np.power(float(10), -8):
                if curve[L-1,0]-curve[0,0]<0 and (curve[L-1, 1]-curve[0,1])>=0:
                    tangent_direction = np.arctan((curve[L-1, 1]-curve[0,1])/(curve[L-1,0]-curve[0,0]))+np.pi
                elif curve[L-1,0]-curve[0,0]<0 and (curve[L-1, 1]-curve[0,1])<0:
                    tangent_direction = np.arctan(
                        (curve[L - 1, 1] - curve[0, 1]) / (curve[L - 1, 0] - curve[0, 0])) - np.pi
                else:
                    tangent_direction = np.arctan((curve[L - 1, 1] - curve[0, 1]) / (curve[L - 1, 0] - curve[0, 0]))
            else:
                # z知道三点找出圆心
                x0 = 1/2*(-y1*np.power(x2,2)+y3*np.power(x2,2)-y3*np.power(y1,2)-y3*np.power(x1,2)-y2*np.power(y3,2)+y1*np.power(x3,2)+y2*np.power(y1,2)-y2*np.power(x3,2)-\
                          y1*np.power(y2,2)+y2*np.power(x1,2)+y1*np.power(y3,2)+y3*np.power(y2,2))/\
                     (-y1*x2+y1*x3+y3*x2+x1*y2-x1*y3-x3*y2)
                y0 = -1/2*(x2*np.power(x1,2)-x3*np.power(x1,2)+x2*np.power(y1,2)-np.power(y1,2)*x3+np.power(x3,2)*x1-\
                           x1*np.power(x2,2)-x2*np.power(x3,2)-x2*np.power(y3,2)+x3*np.power(y2,2)+x1*np.power(y3,2)-\
                           x1*np.power(y2,2)+x3*np.power(x2,2))/(-y1*x2+y1*x3+y3*x2+x1*y2-x1*y3-x3*y2)
                # 轮廓角主方向
                radius_direction = np.arctan((y0-y1)/(x0-x1))
                if (x0-x1)<0 and (y0-y1)>0:
                    radius_direction = radius_direction+np.pi
                elif (x0-x1)<0 and (y0-y1)<0:
                    radius_direction = radius_direction-np.pi
                adjcent_direction = np.arctan((y2-y1)/(x2-x1))
                if (x2-x1)<0 and (y2-y1)>0:
                    adjcent_direction = adjcent_direction+np.pi
                elif (x2-x1)<0 and (y2-y1)<0:
                    adjcent_direction = adjcent_direction-np.pi
                tangent_direction = np.sign(np.sin(adjcent_direction-radius_direction))*np.pi/2+radius_direction

        else:
            tangent_direction = np.arctan((curve[L-1,1]-curve[0,1])/(curve[L-1,0]-curve[0,0]))
            if (curve[L-1,0]-curve[0,0]) < 0 and (curve[L-1,1]-curve[0,1]) > 0:
                tangent_direction = tangent_direction + np.pi
            elif (curve[L-1,0]-curve[0,0]) < 0 and (curve[L-1,1]-curve[0,1]) < 0:
                tangent_direction = tangent_direction - np.pi

        direction = np.append(direction,tangent_direction*180/np.pi)
    ang = np.abs(direction[0]-direction[1])
    # 返回轮廓角，中点到圆心连线的角度
    return ang