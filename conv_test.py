import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np

img = plt.imread("lena512gray.jpg")                 #在这里读取图片

plt.imshow(img)                                     #显示读取的图片
pylab.show()

fil = np.array([[ 0,1,0],                        #这个是设置的滤波，也就是卷积核
                [ 1,0,-1],
                [ 0,-1,0]])

res = cv2.filter2D(img,-1,fil)                      #使用opencv的卷积函数

plt.imshow(res)                                     #显示卷积后的图片
plt.imsave("res.jpg",res)
pylab.show()