from PIL import Image
import os

def TraFilesFor(filelist, formation):
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

def get_imlist(path) -> list:
    '''返回目录中所有JPG图像的文件名列表'''
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def Creatthumb():
    '''使用thumbnail 为test Img创建一个缩略图'''
    img = Image.open(localdir+"\\picture_test\\test.jpg","r")
    img.thumbnail((128,128))
    img.save(localdir + "\\picture_test\\test_thumbnail.jpg")
    #thumnail 接受一个元组
    return

def CropAndPaste():
    '''复制和粘贴图像区域'''
    box = (100,100,400,400)

    return 

def main():
    #todo
    localdir = os.getcwd()
    filelist = get_imlist(localdir + "/picture")

    #print(filelist)
    #TraFilesFor(filelist, ".png")
    #Creatthumb()

    return

if __name__ == '__main__':
    main()