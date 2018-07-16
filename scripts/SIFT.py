#Scale-Invariant Feature Transform
#only for Window or Linux

import numpy as np
from PIL import Image
import os

def process_image(imagename,resultname,params = "--edge-thresh 10 --peak-thresh 5"):
    '''
    处理一幅图像，然后将结果保存到文件中
    '''

    if imagename[-3:] != 'pgm':
        #创建一个Pgm文件
        im = Image.open(imagename).convert('L')
        im.save('temp.pgm')
        imagename = 'temp.pgm'

    cmmd = str("SIFT " + imagename + " --output=" + resultname+
            " " + params)
    
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)

    return

def read_feature_from_file(filename):
    '''
    读取特征值属性，然后将其以矩阵形式返回
    '''

    f = np.loadtext(filename)
    #返回特征位置、描述子
    return f[:,:4], f[:,4:]