#使用環境說明
#darknet目錄位置：/mnt/MIL/neverleave0916/code/darknet
#xml將會移動到：./data/xml
#照片所在位置./data/obj
#xml原先所在位置：./data/obj
#train.txt輸出位置：/data/train.txt
#本檔案應放在./darknet



import glob
import os
import pickle
import shutil
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

classes = ['broiler','crest']
cwd = getcwd() #/mnt/MIL/neverleave0916/code/darknet

xml_path = 'data/xml'
img_path = 'data/obj'
full_xml_path = cwd + '/' + xml_path + '/' #將xml移動到此路徑
full_img_path = cwd + '/' + img_path + '/'
output_path = full_img_path #txt輸出路徑


#輸入的path，加上文件名稱，產生的List
def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path) #返回path最后的文件名
    basename_no_ext = os.path.splitext(basename)[0] #将文件名和扩展名分开

    in_file = open(dir_path + '/' + basename_no_ext + '.xml') #/mnt/MIL/neverleave0916/code/darknet/data/obj /  NVR_ch3_20200512064405_T.xml
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

#輸入xml所在之相對路徑，將xml移動到full_xml_path
def move_xml(img_path):
    if not os.path.exists(full_xml_path):
        os.makedirs(full_xml_path)
    for filename in glob.glob(img_path + '/*.xml'): #移動xml檔案
        shutil.move(filename,full_xml_path)




if not os.path.exists(output_path):
    os.makedirs(output_path)

image_paths = getImagesInDir(img_path)           #輸入'data/obj' 返回 data/obj/img1.jpg 的List
list_file = open(cwd + '/data/train.txt', 'w')   #開啟/mnt/MIL/neverleave0916/code/darknet/data/train.txt
for image_path in image_paths:
    list_file.write(image_path + '\n')
    convert_annotation(full_img_path, output_path, image_path)
    
list_file.close()
move_xml(img_path)
print("Finished processing: " + img_path)