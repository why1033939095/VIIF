# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from PIL import Image
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import tkinter
import math
import cv2

from py_resizeImage import py_resizeImage
from py_registration import py_registration
from py_getAffine import py_getAffine
from py_subpixelFine import py_subpixelFine
from py_graymosaic import py_graymosaic
from py_showResult import py_showResult
from py_showMatch import py_showMatch
from py_rgbmosaic import py_rgbmosaic
from readTxt import readTxt






def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # 获取红外和可见光的灰度图，rgb图
    # IRgray, VIgray, IRrgb, VIrgb = py_readImage()
    file_path = r'C:\Users\WHY\Desktop\公司文件\Image-registration-master\Image-registration-master\CAO-C2F\Example_Images'

    I1rgb = np.array(plt.imread(file_path+'\I1.jpg'))

        # 获取灰度图，用cv2的函数直接获得，转化为浮点类型
        # IR_gray = cv2.imread(file_path, 0)
    I1gray = Image.open(file_path+'\I1.jpg')
    I1gray = np.array(I1gray.convert('L'))
    # I1gray = np.array(Image.open('./image/I1_m.png'))



    I2rgb = np.array(plt.imread(file_path+'\V1.jpg'))
        # VI_gray = cv2.imread(file_path, 0)
    I2gray = Image.open(file_path+'\V1.jpg')
    I2gray = np.array(I2gray.convert('L'))



    # 令整个图像最下像素值为0，最大像素值为255
    I1gray = np.round((np.array(I1gray) - np.array(I1gray).min()) / (np.array(I1gray).max() - np.array(I1gray).min()) * 255)

    I2gray = np.round((np.array(I2gray) - np.array(I2gray).min()) / (np.array(I2gray).max() - np.array(I2gray).min()) * 255)

    print('【0】Completed to read image!')
    # 获得图片的高
    height = I1gray.shape[0]

    # scale为2x1的矩阵，为1和I2和I1的高度比；I1和I2大小相同
    I1, I2, scale = py_resizeImage(I1gray, I2gray, height)
    # I2 = np.array(Image.open('./image/I2_m.png'))
    I1_itea = I1
    # 迭代次数
    iterationNum = 1
    iteration = 0
    # 4*红外图的高度（最大出错的点的个数） I2的高度除以300再乘以4，向上取整，RMSE是均方根误差
    maxRMSE = 4*math.ceil(I2.shape[0]/300)
    # 3x3x1的全零数组
    AffineTrans = np.zeros((3, 3, iterationNum))


    while iteration < iterationNum:
        print(f'\n{iteration}(th) iteration of registration...\n')
        # corner_iv是[[IR中点坐标xy互换],[0,0],[VI中点坐标xy互换]]
        P1, P2, corner12 = py_registration(I1_itea, I2, 20, maxRMSE, iteration, 1, 0, 6, 1, I2gray)
        # P1 = readTxt(r'.\image\p1.txt', splt='\t')
        # P2 = readTxt(r'.\image\p2.txt', splt='\t')
        I1_itea, affmat = py_getAffine(I1_itea, I2, P1, P2)
        # affmat = np.append(affmat, np.array([[0, 0, 1]]), axis=0)
        AffineTrans[:,:,iteration] = affmat.T
        iteration += 1


    P1 = np.append(P1, np.ones((P1.shape[0],1)), axis=1)
    # 第一列中等于零的坐标的序号（y坐标），就是分开cor1和cor2的分界线
    pos_cor1 = np.where(corner12[:,0] == 0)[0]
    for i in range(iteration-1,0,-1):
        P1 = np.dot(P1,np.linalg.pinv(AffineTrans[:,:,iteration]))
        # 把第一列中等于零的坐标删掉
        cor12 = np.dot(np.append(np.array(corner12[0:pos_cor1[0],0:2]), np.ones((pos_cor1[0],1))), np.linalg.pinv(AffineTrans[:,:,i-1]))
        # 仿射后前两列除以第三列
        P1[:,0:2] = P1[:,0:2] / P1[:,2]
        P1[:,2] = np.ones((P1.shape[0],1))
        corner12[0:pos_cor1[0],0:2] = cor12[:,0:2]/cor12[:,2]
        corner12[0:pos_cor1[0],2] = np.ones(pos_cor1[0],1)
    P1 = P1[:,0:2]
    corner12 = corner12[:,0:2]
    # 在原图中的正确匹配
    # 对图进行缩放
    P1[:,1] = I1gray.shape[0]/2+scale[0]*(P1[:,1]-I1.shape[0]/2)
    P1[:,0] = I1gray.shape[1]/2+scale[0]*(P1[:,0]-I1.shape[1]/2)
    # 往左上平移图片
    corner12[0:(np.round(pos_cor1[0])).astype(int),1] = I1gray.shape[0]/2+scale[0]*(corner12[0:(np.round(pos_cor1[0])).astype(int),1]-I1.shape[0]/2)
    corner12[0:(np.round(pos_cor1[0])).astype(int),0] = I1gray.shape[1]/2+scale[0]*(corner12[0:(np.round(pos_cor1[0])).astype(int),0]-I1.shape[1]/2)

    P2[:,1] = I2gray.shape[0]/2+scale[1]*(P2[:,1]-I2.shape[0]/2)
    P2[:,0] = I2gray.shape[1]/2+scale[1]*(P2[:,0]-I2.shape[1]/2)
    corner12[(np.round(pos_cor1[0])).astype(int)+1:corner12.shape[0], 1] = I2gray.shape[0]/2+scale[1]*(corner12[(np.round(pos_cor1[0])).astype(int)+1:corner12.shape[0],1]-I2.shape[0]/2)
    corner12[(np.round(pos_cor1[0])).astype(int)+1:corner12.shape[0], 0] = I2gray.shape[1]/2+scale[1]*(corner12[(np.round(pos_cor1[0])).astype(int)+1:corner12.shape[0],0]-I2.shape[1]/2)


    # print('P1:\n',P1)
    # print(('P2:\n',P2))
    P3 = py_subpixelFine(P1,P2)
    _,affmat = py_getAffine(I1gray,I2gray,P1, P3)
    Imosaic = py_graymosaic(I1gray,I2gray, affmat)
    plt.imshow((py_rgbmosaic(I1rgb, I2rgb, affmat)).astype(int))
    plt.show()

    py_showResult(I1rgb, I2rgb, I1gray, I2gray, affmat, 3)
    py_showMatch(I1rgb,I2rgb,P1,P2,[],'before')
    py_showMatch(I1rgb,I2rgb,P1,P3,[],'after')












