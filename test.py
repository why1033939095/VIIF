from readTxt import readTxt
import numpy as np
from PIL import Image


file_path = r'C:\Users\WHY\Desktop\公司文件\Image-registration-master\Image-registration-master\CAO-C2F\Example_Images'


# 获取灰度图，用cv2的函数直接获得，转化为浮点类型
# IR_gray = cv2.imread(file_path, 0)
I1gray = Image.open(file_path + '\I1.jpg')
I1gray = np.array(I1gray.convert('L'))
I1 = np.round((np.array(I1gray) - np.array(I1gray).min()) / (np.array(I1gray).max() - np.array(I1gray).min()) * 255)

I2gray = Image.open(file_path + '\V1.jpg')
I2gray = np.array(I2gray.convert('L'))
I2 = np.round((np.array(I2gray) - np.array(I2gray).min()) / (np.array(I2gray).max() - np.array(I2gray).min()) * 255)

r1, c1 = I1.shape
r2, c2 = I2.shape

# 呈左上右下进行排列图片
Imosaic = np.zeros((r2 + 2 * np.max((r1, c1)), c2 + 2 * np.max((r1, c1))))

affmat = np.array([[0.1899,1.2083,-234.1215],[-1.1379,0.1529,776.0227],[1.2002e-04,-1.4883e-05,1]])

affinemat = affmat.T
# 找出不是数字的部分，即每个点的x和y坐标
u, v = np.where(~ np.isnan(Imosaic.T))
# 每个点的减去max(图1长宽)
v = v - np.max((r1, c1)) + 1
u = u - np.max((r1, c1)) + 1

# x坐标，y坐标，1
utvt = np.dot(np.append(np.append(np.array([u]).T, np.array([v]).T, axis=1), np.ones((v.shape[0], 1)), axis=1),
              np.linalg.pinv(affinemat))
ut = utvt[:, 0] / utvt[:, 2]
vt = utvt[:, 1] / utvt[:, 2]
utu = np.reshape(ut, (c2 + 2 * np.max((r1, c1)), r2 + 2 * np.max((r1, c1)))).T
vtv = np.reshape(vt, (c2 + 2 * np.max((r1, c1)), r2 + 2 * np.max((r1, c1)))).T

y = utu-1
x = vtv-1

a = I1

flag = (x<=a.shape[0]-1) * (x>=0) * (y<=a.shape[1]-1) * (y>=0)

un,vn = np.where(flag.T)
vmin1 = np.min(vn)
vmax1 = np.max(vn)
umin1 = np.min(un)
umax1 = np.max(un)

Imosaic[np.max((r1,c1)):np.max((r1,c1))+r2,np.max((r1,c1)):np.max((r1,c1))+c2] = I2

flag2 = np.zeros_like(Imosaic)
flag2[np.max((r1,c1)):np.max((r1,c1))+r2,np.max((r1,c1)):np.max((r1,c1))+c2] = np.full((np.max((r1,c1))+r2-np.max((r1,c1)),np.max((r1,c1))+c2-np.max((r1,c1))), 1)

# 进行二维线性插值
x = x*flag
y = y*flag
t00 = a[np.floor(x).astype(int), np.floor(y).astype(int)]
t10 = a[np.ceil(x).astype(int), np.floor(y).astype(int)]
t01 = a[np.floor(x).astype(int), np.ceil(y).astype(int)]
t11 = a[np.ceil(x).astype(int), np.ceil(y).astype(int)]

x0 = np.ceil(x)-x
x1 = x-np.floor(x)
y0 = np.ceil(y)-y
y1 = y-np.floor(y)

b0 = t00*y0+t01*y1
b1 = t10*y0+t11*y1
c = b0*x0+b1*x1
c = c*flag


Imosaic[np.where(flag==True)] = ((c*2+flag*flag2*(Imosaic-c))/2)[np.where(flag==True)]
validuv = np.array([[np.min((vmin1+1,np.max((r1,c1)))), np.min((umin1+1,np.max((r1, c1))))],[np.max((vmax1+1,np.max((r1,c1))+r2)), np.max((umax1+1, np.max((r1,c1))+c2))]])
Imosaic = Imosaic[validuv[0][0]-1:validuv[1][0], validuv[0][1]-1:validuv[1][1]].astype(np.uint8)

print('ok')




















