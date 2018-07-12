import numpy as np 
import pylab as pl 
import os
from PIL import Image
from scipy.ndimage import filters

def compute_harris_response(im,sigma = 3):
    '''在一幅灰度图像中，对每个像素计算Harris角点检测器'''

    #计算导数
    imx = np.zeros(im.shape)
    imy = np.zeros(im.shape)
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

def get_harris_points(harrisim, min_dist = 10, threshold = 0.05):
    '''从一幅Harris响应图像中 返回角点，min_dist 为分割角点'''

    #寻找高于阈值的候选角点
    corner_threshold = harrisim.max()* threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    #得到候选点的坐标
    coords = np.array(harrisim_t.nonzero()).T

    #以及它们的Harris响应值
    candidate_values = [harrisim[c[0],c[1]] for c in coords]

    #对候选点按照Harris 响应值进行排序
    index = np.argsort(candidate_values)

    #将可行点的位置保存到数组中
    allowed_locations = np.ones(harrisim.shape)
    #allowed_locations[min_dist : -min_dist, min_dist : -min_dist] = 1
    #print(allowed_locations)

    #按照min_distance原则，选择最佳Harris 点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[1,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0] - min_dist):(coords[1,0] + min_dist)
            ,(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    print(filtered_coords)
    return filtered_coords

def plot_harris_points(img, filtered_coords):
    '''
    绘制图像中的检测点
    '''
    pl.figure()
    pl.gray()
    pl.imshow(img)
    pl.plot([p[1] for p in filtered_coords],
    [p[0] for p in filtered_coords],"*")
    pl.show()
    return 

def get_descriptors(image,filtered_coords,wid = 5):
    '''
    对于每个返回的点，返回点周围 2*wid +1个像素 的值
    '''    
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
            coords[1] - wid : coords[1] + wid +1].flatten()
        desc.append(patch)
    
    return desc

def match(desc1,desc2,threshold = 0.5):
    '''
    对于第一幅图像中的每个角点描述子，使用归一化互相关
    选取它在第二幅图像中的匹配角点
    '''
    
    n = len(desc1[0])

    #点对的距离
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(Desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2)/(n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
    
    ndx = argsort(-d)
    matchscores = ndx[:,0]

    return matchscores

def appendimages(im1,im2):
    '''
    返回将两幅图像并排拼接的一幅新的图像

    '''
    #TODO

def main():
    cd = os.path.dirname(os.getcwd())
    img = Image.open(cd + "\\PCVwithPython\\picture_test\\test.jpg","r")
    im = np.array(img.convert('L'))
    harrisim = compute_harris_response(im)
    filtered_coords = get_harris_points(harrisim,6)
    print(filtered_coords)
    plot_harris_points(im,filtered_coords)

    return 

if __name__ == '__main__':
    main()