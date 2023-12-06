import cv2
import numpy as np
import matplotlib.pyplot as plt



def py_showResult(I1rgb, I2rgb, I1gray, I2gray, aff, checkerboard):
    #显示配准结果，原图，融合图，仿射图
    I1_aff = np.zeros_like(I2rgb)
    aff = aff[0:2]
    if I1rgb.shape[2] == 3 and I2rgb.shape[2] == 3:
        I1_aff[:,:,0] = cv2.warpAffine(np.float32(I1rgb[:,:,0]), np.float32(aff), (I2gray.shape[1],I2gray.shape[0]))
        I1_aff[:,:,1] = cv2.warpAffine(np.float32(I1rgb[:,:,1]), np.float32(aff), (I2gray.shape[1],I2gray.shape[0]))
        I1_aff[:,:,2] = cv2.warpAffine(np.float32(I1rgb[:,:,2]), np.float32(aff), (I2gray.shape[1],I2gray.shape[0]))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.subplot(121)
        plt.imshow(I1rgb)
        plt.title('Source IR image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(I2rgb)
        plt.title('Source VI image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        I1_aff = cv2.warpAffine(I1gray, aff, I2gray.shape)
        I1_aff = float(I1_aff)
        I1_aff = I1_aff/np.max(I1_aff)
        I2gray = float(I2gray)
        I2gray = I2gray/np.max(I2gray)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
        plt.subplot(221)
        plt.imshow(I1gray)
        plt.title('Source image')
        plt.axis('off')

        plt.subplot(222)
        plt.imshow(I2gray)
        plt.title('Source image')
        plt.axis('off')

        plt.subplot(223)
        plt.imshow(I1_aff)
        plt.title('I1 transformation')
        plt.axis('off')

        plt.subplot(224)
        plt.imshow(I1_aff+I2gray)
        plt.title('Gray-level Everage Fusion')
        plt.axis('off')

        # 设置子图默认的间距
        plt.tight_layout()
        #显示图像
        plt.show()