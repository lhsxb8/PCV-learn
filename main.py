from PIL import Image
import pylab as pl
import os
import numpy as np
from scipy.ndimage import filters
import imtool

localdir = os.getcwd()
class PILtest:
    def TraFilesFor(self,filelist, formation):
        '''读取filelist（文件名列表）里所有图像文件并转化成JPEG格式'''
        for infile in filelist:
            outfile = os.path.splitext(infile)[0] + formation
            #splitext用于分离文件名和拓展名，返回一个包含字符串的元组
            if infile != outfile:
                try:
                    Image.open(infile).save(outfile)
                except IOError:
                    print("cannot convert" + infile)
        return

    def get_imlist(self,path) -> list:
        '''返回目录中所有JPG图像的文件名列表'''
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

    def Creatthumb(self,img):
        '''使用thumbnail 为test Img创建一个缩略图'''
        
        img.thumbnail((128,128))
        img.save(localdir + "\\picture_test\\test_thumbnail.jpg")
        #thumnail 接受一个元组
        return

    def CropAndPaste(self,img):
        '''复制和粘贴图像区域'''
        box = (100,100,400,400)
        #四元组坐标依次（左，上，右，下）
        #PIL指定左上角为（0,0）

        region = img.crop(box)
        region = region.transpose(Image.ROTATE_180)
        img.paste(region,box)
        img.save(localdir + "\\picture_test\\test_2.jpg")
        return 

    def ResizeAndRotate(self,img,newsize=(128,128),rotation=90):
        '''Resize the image to (row,col),and rotate'''
        out = img.resize(newsize)
        out = img.rotate(rotation)
        out.save(localdir + "\\picture_test\\test_3.jpg")
        return

    #def 

class PylabTest():
    def GetImgToArray(self,img):
        '''将图片读取到数组中'''
        im = np.array(img)
        print(im)
        return im
    
    def ShowImage(self,img):
        '''显示图像'''
        pl.imshow(img)
        return
    
    def DrawSomePlots(self,img=None):
        '''画一些点'''
        pl.imshow(img)

        x=[100,120,300,300]
        y=[200,500,200,300]
        pl.plot(x,y,'r*')
        '''
        基本颜色命令
        'b'-> blue      'g'->green  'r'->red    'c'->cyan   'm'->magenta
        'y'-> yellow    'k'->black  'w'->white   
        plot(x,y)      默认为蓝色实线
        plot(x,y,'r*') 红色星状标记
        plot(x,y,'go-')带有圆圈标记的绿线
        plot(x,y,'ks:')带有正方形标记的黑色点线
        '''
        #绘制连接前两个点的线
        pl.plot(x[:2],y[:2],'ks:')

        #这个命令可以让坐标轴消失
        pl.axis('off')

        #添加标题
        pl.title('Plotting:"test.jpg"')

        #show只能调用一次，通常放在脚本的结尾
        pl.show()
        return

    def PictureContour(self,img):
        '''图像轮廓'''
        #将图像转化为灰度图
        im = np.array(img.convert('L'))

        #新建一个图像
        pl.figure()

        #不使用颜色信息
        pl.gray()

        #在原点的左上角显示轮廓图像
        pl.contour(im,origin="image")
        pl.axis('equal')
        pl.axis('off')
        pl.show()
        return

    def DrawHist(self,img):
        '''画直方图'''

        im = np.array(img.convert('L'))
        pl.figure()
        #画直方图
        #hist 函数可以绘制直方图
        #hist()只接受一位数组，第二个参数指定小区间的数目
        pl.hist(im.flatten(),128)#flatten可以将任意数组按照行优先准则转化成一维数组
        pl.show()
        return

class NumpyTest():
    def NpArraytest(self,img):
        '''np.array的使用'''
        im = np.array(img)
        print(im.shape,im.dtype,flush=False)
        
        im = np.array(img.convert('L'),'f')
        print(im.shape,im.dtype,flush=False)
        
        #第一个元组表示图像数组的大小（行、列、颜色通道）
        #第二个字符串表示数组元素的数据类型
        #图像通常被编码成无符号八位整数（uint8)
        #第二行在创建数组时已经使用了额外的参数'f' ，将数据类型转化为浮点型
        #因为灰度图没有颜色信息，所以在形状元组中只有两个数值

        #获取数组中的元素像素值 坐标（20,30）
        #第一通道im[20,30,1]
        value = im[20,30]
        print(value)
        #关于灰度图像的切片方式访问
        im[5,:] = im[10,:]  #将第10行的数值赋值给第5行
        im[:,5] = 100
        temp0 = im[:100,:50].sum()        #计算前100行，前50列的所有数值的和
        temp1 = im[50:100,50:100].sum()
        temp2 = im[50].mean()        #第50行所有数值的平均值
        temp3 = im[:,-1]            #最后一列
        temp4 = im[-2,:]            #倒数第二行 orim[-2]
        return

    def GrayLVtransf(self,img):
        im = np.array(img.convert('L'))

        pl.figure()
        #对图像进行反相处理
        im2 = 255 - im
        img2 = Image.fromarray(im2)
        #if has erroe: 'numpy.ndarray' object has no attribute 'mask'
        #you need reinstall your matplotlib
        #2.1.2=>2.2.2
        pl.imshow(img2)

        pl.figure()
        #对图像灰度值压缩在100-200
        im3 = (im/255)*100+100
        img3 = Image.fromarray(im3)
        print(im3)
        pl.imshow(img3)

        pl.figure()
        #对图像进行平方处理
        im4 = 255*(im/255)**2
        img4 = Image.fromarray(im4)
        pl.imshow(img4)

        pl.show()
        return

    def CDFHistAverage(self,img):
        '''
        直方图均衡化，
        将一幅图像的灰度直方图变平，使每个灰度值的分布概率相同
        '''
        pl.figure()
        im = np.array(img.convert('L'))
        im2,cdf = imtool.histeq(im)
        img2 = Image.fromarray(im2)
        pl.imshow(img2)
        pl.show()
        return

class ScipyTest():
    def GrayGuassianFliter(self,img,a):
        '''灰度图高斯模糊测试'''
        pl.figure
        im = np.array(img.convert('L'))
        im2 = filters.gaussian_filter(im,a)
        img = Image.fromarray(im2)
        pl.imshow(img)
        pl.show()
        return

    def ColorGuassianFliter(self,img,a):
        '''
        彩色图高斯模糊测试
        可以看到,a越大,模糊程度越大
        '''
        pl.figure()
        im = np.array(img)
        im2 = np.zeros(im.shape)
        for i in range(3):
            im2[:,:,i] = filters.gaussian_filter(im[:,:,i],a)
            #其中的im2[:,:,i]相当于把所有的子矩阵的第i维提取出来
            #在这里就是等于所有像素值的第i个通道值
        im2 = np.uint8(im2)
        img = Image.fromarray(im2)
        pl.imshow(img)
        pl.show()
        return



def main():
    #todo
    img = Image.open(localdir+"\\picture_test\\test.jpg","r")
    '''Image API test'''
    #test1 = PILtest()
    #filelist = test1.get_imlist(localdir + "\\picture")
    #print(filelist)
    #test1.TraFilesFor(filelist, ".png")
    #test1.Creatthumb(img)
    #test1.CropAndPaste(img)
    #test1.ResizeAndRotate(img)

    '''Matlop API test'''
    #test2 = PylabTest()
    #test2.DrawSomePlots(img)
    #test2.PictureContour(img)
    #test2.DrawHist(img)
    
    '''Numpy API test'''
    #test3 = NumpyTest()
    #test3.NpArraytest(img)
    #test3.GrayLVtransf(img)
    #test3.CDFHistAverage(img)


    '''Scipy API test'''
    test4 = ScipyTest()
    test4.ColorGuassianFliter(img,5)
    return

if __name__ == '__main__':
    main()