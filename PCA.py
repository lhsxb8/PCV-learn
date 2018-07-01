from PIL import Image
import numpy as np

'''
Principal Component Analysis 组成成分分析
'''
def Pca(X):
    '''
    输入：矩阵X，其中该矩阵中存储训练模型数据，每一行为一条训练数据
    返回：投影矩阵（按照维度的重要性排序）、方差、均值
    '''

    #获取维度
    num_data,dim = X.shape

    #数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:
        #PCA-使用紧致技巧
        M = np.dot(X,X.T)   #dot()矩阵相乘、XXT为协方差矩阵
        #https://blog.csdn.net/lifeng_math/article/details/50014073
        #贴一个非常详细的讲解博客
        #维基百科详解协方差矩阵
        #https://en.wikipedia.org/wiki/Covariance_matrix

        e, EV = np.linalg.eigh(M)   #获得M矩阵的特征值和特征向量
        tmp = dot(X.T,EV).T     #紧致技巧
        V = tmp[::-1]   #倒序:最后的特征向量才是需要的，将其逆转
        S = sqrt(e)[::-1]   #特征值也要逆转
        for i in range (V.shape[1]):#shape[0]是行数，shape[1]是列数
            V[:,i] /= S

    else:
        #PCA 使用SVD方法
        U,S,V = np.linalg.svd(X)
        V = V[:num_data]

return V,S,mean_X



    return 