# -*- coding: utf-8 -*-
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from skimage import io,draw

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def cvDrawBoxes2(detections, image):
    imcaption = []
    for detection in detections:
        label = detection[0]
        confidence = detection[1]
        bounds = detection[2]
        shape = image.shape
        # x = shape[1]
        # xExtent = int(x * bounds[2] / 100)
        # y = shape[0]
        # yExtent = int(y * bounds[3] / 100)
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        # Coordinates are around the center
        xCoord = int(bounds[0] - bounds[2]/2)
        yCoord = int(bounds[1] - bounds[3]/2)
        boundingBox = [
            [xCoord, yCoord],
            [xCoord, yCoord + yExtent],
            [xCoord + xEntent, yCoord + yExtent],
            [xCoord + xEntent, yCoord]
        ]
        rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
        rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
        rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
        rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
        rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
        boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
        draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
        draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
        draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
        draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
        draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)

        
        
        
        
        x, y, w, h = detection[2][0],\
        detection[2][1],\
        detection[2][2],\
        detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.putText(image,
            detection[0].decode() +
            " [" + str(round(detection[1] * 100, 2)) + "]",
            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            [0, 255, 0], 2)
    cv2.putText(image,str(len(detections))+" Results",(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2) 
    return image


def get_coordinate(detections):
    coordinate = []
    for detection in detections:
        coordinate.append((detection[2][0],detection[2][1]))
    return coordinate
    #print(coordinate)


netMain = None
metaMain = None
altNames = None


# +
def decode_fourcc(v):
  v = int(v)
  return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

#        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
#        codec = decode_fourcc(fourcc)
#        print("Codec: " + codec)


# -

def YOLO():

    global metaMain, netMain, altNames
    configPath = "yolo-obj.cfg"
    weightPath = "backup/yolo-obj_best.weights"
    metaPath = "data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test/test3.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    #out = cv2.VideoWriter(
    #    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #    (darknet.network_width(netMain), darknet.network_height(netMain)))
    out = cv2.VideoWriter("test/output2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
          (int(cap.get(3)),int(cap.get(4))))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    #darknet_image = darknet.make_image(darknet.network_width(netMain),
    #                                darknet.network_height(netMain),3)
    darknet_image = darknet.make_image(int(cap.get(3)),int(cap.get(4)),3)
    while (cap.isOpened()):
        prev_time = time.time()
        ret, frame_read = cap.read()
        

        if ret is True:
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            #frame_rgb = cv2.resize(frame_rgb,(720,480))
            #frame_resized = cv2.resize(frame_rgb,
            #                           (darknet.network_width(netMain),darknet.network_height(netMain)),
            #                           interpolation=cv2.INTER_LINEAR)
            #darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            darknet.copy_image_from_bytes(darknet_image,frame_rgb.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25) #把圖片拿去測試
            #image = cvDrawBoxes(detections, frame_resized)
            #image = cvDrawBoxes(detections, frame_rgb)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print(1/(time.time()-prev_time))

            #print(get_coordinate(detections)) 取得座標的List
            print("***************" + str(len(detections)) + "Results***************")
            #for elem in detections:
                #print ("%.3f,%.3f" % (elem[2][0]+(elem[2][2]/2),elem[2][1]+(elem[2][3]/2)))
                #print ("%.1f,%.1f" % (elem[2][0],elem[2][1])) #輸出座標
                #print(elem[2])
            #print("\n\n\n")
            image = cvDrawBoxes2(detections, frame_rgb) #畫框框
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #轉換顏色
            out.write(image)
            cv2.imshow('Demo', image)
            #cv2.waitKey(3) #若設為0，等待按鍵輸入後才會下一偵
            if cv2.waitKey(1) & 0xFF == ord('q'): #按下q可以結束
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    YOLO()
