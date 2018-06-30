from PIL import Image
import numpy as np
import os
import pylab as pl

def get_imlist(self,path) -> list:
        '''返回目录中所有JPG图像的文件名列表'''
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def histeq(img,nbr_bins=256):
    '''
    对图像进行直方图均衡化
    input:
        img:array(灰度图像)
        nbr_bins:使用小区间的数目
    '''
    imhist,bins = np.histogram(img.flatten(),nbr_bins,normed=True)
    #histogram() function return two para:
    #1. hist:the val of histogram
    #2. bin_edges:array of dtype float (
    # return bin edges(length(hist)+1)
    # )

    cdf = imhist.cumsum()#cumulative distribution function =>cdf
    #分布函数（类似）
    cdf = 255 * cdf / cdf[-1] #归一化 
    #将像素值映射到目标范围你的归一操作
    
    #使用累计分布函数的线性插值，计算新的像素值
    im2 = np.interp(img.flatten(),bins[:-1],cdf)#bins[:-1]=>去除最后一个数据后的数组

    return im2.reshape(img.shape),cdf    #将im2数组重新eshape成原来格式

def compute_average(imlist):
    '''计算图像列表的的平均图像'''

    #打开第一幅图像，将其存储在浮点型数组中
    averageim = array(Image.open(imlist[0]),'f')

    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + "can't open")
        averageim /= len(imlist)

    #返回uint8类型的平均图像
    #图像一般存储成uint8类型！
    return array(averageim, 'uint8')