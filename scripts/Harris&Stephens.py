import numpy as np 
import pylab as py 
import os
from PIL import Image
from scipy.ndimage import filters

def compute_harris_response(im,sigma = 3):
    '''在一幅灰度图像中，对每个像素计算Harris角点检测器'''

    #计算导数
    imx = zeros(im.shape())
    imy = zeros(im.shape())
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

    #计算Harris 矩阵的分量
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    #计算特征值和迹
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet/Wtr

    return 

def main():
    cd = os.path.dirname(os.getcwd())
    file = Image.open(cd + "\\picture_test\\test.jpg","r")
    im = np.array(file)

    return 

if __name__ == '__main__':
    main()