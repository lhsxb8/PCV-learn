import numpy as np 
import time
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
    #print(Wdet/Wtr)

    return Wdet/Wtr


def get_harris_points(harrisim, min_dist = 1, threshold = 0.05):
    '''从一幅Harris响应图像中 返回角点，min_dist 为分割角点'''

    #寻找高于阈值的候选角点
    corner_threshold = harrisim.max()* threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    #得到候选点的坐标
    coords = np.array(harrisim_t.nonzero()).T
    #print(coords.shape[0])
    #print(coords)
    #以及它们的Harris响应值
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    #对候选点按照Harris 响应值进行排序
    index = np.argsort(candidate_values,axis=0)
    print(index)
    #将可行点的位置保存到数组中
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist , min_dist:-min_dist] = 1
    print(allowed_locations)

    #按照min_distance原则，选择最佳Harris 点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0] + min_dist),
            (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    #print(filtered_coords)
    return filtered_coords

def plot_harris_points(img, filtered_coords):
    '''
    绘制图像中的检测点
    '''
    pl.figure()
    pl.gray()
    pl.imshow(img)
    pl.plot([p[1] for p in filtered_coords],
    [p[0] for p in filtered_coords],"r*")
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
    d = -np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2)/(n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
    
    ndx = np.argsort(-d)
    matchscores = ndx[:,0]

    return matchscores

def match_twosided(desc1,desc2,threshold = 0.5):
    '''
    两边对称的match
    '''
    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)

    ndx_12 = np.where(matches_12 >= 0)[0]

    #去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12

def appendimages(im1,im2):
    '''
    返回将两幅图像并排拼接的一幅新的图像

    '''
    
    #首先要确保两张图像的行数相同
    #对较少行数的图像进行填充

    row1 = im1.shape[0]
    row2 = im2.shape[0]

    if row1 < row2:
        im1 = np.concatenate((im1,np.zeros((row2 - row1,im1.shape[1]))),axis = 0)
    elif row2 < row1:
        im2 = np.concatenate((im2,np.zeros((row2 - row1,im2.shape[1]))),axis = 0)

    #如果行数相同，直接返回

    return np.concatenate((im1,im2),axis = 1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below = True):
    '''
    显示一幅带有连接匹配之间连线的图片
    输入：im1,im2 数组图像 locs1,locs2 特征位置 matchscores(match()的输出)
    show_below 图像是否应该显示在下方
    '''

    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))
        # np.vstack 垂直堆数组
    
    pl.imshow(im3)

    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m > 0:
            pl.plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    pl.axis('off')


def main():
    time_start = time.time()
    cd = os.path.dirname(os.getcwd())
    print(cd)
    img1 = Image.open(cd + "\\PCVwithPython\\picture_test\\test.jpg","r")
    img2 = Image.open(cd + "\\PCVwithPython\\picture_test\\test.jpg","r")
    im1 = np.array(img1.convert('L'))
    im2 = np.array(img2.convert('L'))


    wid = 5
    harrisim = compute_harris_response(im1,sigma=5)
    filtered_coords_1 = get_harris_points(harrisim,min_dist=10,threshold=0.05)
    #print(filtered_coords)
    #plot_harris_points(im,filtered_coords)
    d1 = get_descriptors(im1,filtered_coords_1,wid)
    
    harrisim = compute_harris_response(im2,sigma=5)
    filtered_coords_2 = get_harris_points(harrisim,wid+1)
    d2 = get_descriptors(im2,filtered_coords_2,wid)

    print('Start Matching')
    matches = match_twosided(d1,d2)

    pl.figure()
    pl.gray()
    plot_matches(im1,im2,filtered_coords_1,filtered_coords_2,matches)
    pl.show()

    time_end = time.time()
    print(time_end - time_start)
    return 

if __name__ == '__main__':
    main()