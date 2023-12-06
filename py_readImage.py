import tkinter
from tkinter import filedialog
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2

DistortFlag = 0

def py_readImage():
    # 获取当前文件夹的的路径
    mainPath = os.getcwd()
    # 创建一个tkinter.Tk()实例
    root = tkinter.Tk()
    # 将tkinter.Tk()实例隐藏
    root.withdraw()
    # 打开对话框时的默认路径
    default_dir = r"C:\Users\PG\Downloads\Image-registration-master\Image-registration-master\CAO-C2F\Example_Images"
    # 显示对话框，选择红外图片，返回图片路径
    file_path = tkinter.filedialog.askopenfilename(title=u'选择红外图片', initialdir=(os.path.expanduser(default_dir)))
    # 得到一个具有红外图相关信息的数组
    try:
        IR_rgb = np.array(plt.imread(file_path))

        # 获取灰度图，用cv2的函数直接获得，转化为浮点类型
        # IR_gray = cv2.imread(file_path, 0)
        IR_gray = Image.open(file_path)
        IR_gray = np.array(IR_gray.convert('L'))

        # 去畸变
        if DistortFlag:
            py_undistorting(IR_rgb)
            py_undistorting(IR_gray)

    except Exception as e:
        print('获取红外图捕获到异常', e)

    # plt.imshow(IR_gray, cmap='gray')
    # plt.show()

    # 可见光图片
    file_path = tkinter.filedialog.askopenfilename(title=u'选择可见光图片', initialdir=(os.path.expanduser(default_dir)))
    try:
        VI_rgb = np.array(plt.imread(file_path))
        # VI_gray = cv2.imread(file_path, 0)
        VI_gray = Image.open(file_path)
        VI_gray = np.array(VI_gray.convert('L'))

        # 去畸变
        if DistortFlag:
            py_undistorting(VI_rgb)
            py_undistorting(VI_gray)
    except Exception as e:
        print('获取可见光图捕获到异常', e)


    IR_gray = (np.array(IR_gray)-np.array(IR_gray).min())/(np.array(IR_gray).max()-np.array(IR_gray).min())*255

    VI_gray = (np.array(VI_gray) - np.array(VI_gray).min()) / (np.array(VI_gray).max() - np.array(VI_gray).min()) * 255

    print('【0】Completed to read image!')
    return IR_gray,VI_gray,IR_rgb,VI_rgb

def py_undistorting(img):
    '''
    使用张正友畸变矫正法：去除畸变，畸变参数，内参矩阵
    具体参数计算参考http://t.csdn.cn/YMDrF
    :param img: 畸变图像
    :return:
    '''
    # 相机内参
    P = [[458.654, 0, 367.215],
         [0, 457.296, 248.375],
         [0, 0, 1]]


    # 畸变参数[k1,k2,p1,p2,k3=None,k4=None,k5=None,k6=None]
    K = [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]

    # cv2去除畸变
    img_distort = cv2.undistort(img, np.array(P), np.array(K))
    # 求出畸变前后图像差异 不需要再插值了
    img_diff = cv2.absdiff(img, img_distort)

    return img_distort


if __name__ == '__main__':
    py_readImage()