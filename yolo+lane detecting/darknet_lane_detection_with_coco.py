from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
def imageCopy(src):
    return np.copy(src)
    
    
def convertColor(image, flag=None):
    if flag is None:
        return image
    else:
        return cv2.cvtColor(image, flag)


def rangeColor(image, lower, upper):
    return cv2.inRange(image, lower, upper)


def splitColor(image, lower, upper, flag=None):
    converted = convertColor(image, flag)
    mask = rangeColor(converted, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)


def polyROI(image, points):
    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]
    mask = np.zeros_like(image)
    ignore_mask_color = (255,) * channels
    cv2.fillPoly(mask, points, ignore_mask_color)
    return cv2.bitwise_and(image, mask)


def convertColor(image, flag=None):
    if flag is None:
        return image
    else:
        return cv2.cvtColor(image, flag)


def cannyEdge(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)


def addWeightedImage(image1, w1, imagw2, w2=None):
    if w2 is None:
        return cv2.addWeighted(image1, float(w1) * 0.01, imagw2, float(100 - w1) * 0.01, 0)
    else:
        return cv2.addWeighted(image1, w1 * 0.01, imagw2, w2 * 0.01, 0)


def houghLinesP(image, rho=1.0, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=100):
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)


def drawHoughLinesP(image, lines):
    result = imageCopy(image)
    if len(image.shape) == 2:
        result = convertColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return result


def splitLines(lines):
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for line in lines:
        x1 = line[0,0]
        y1 = line[0,1]
        x2 = line[0,2]
        y2 = line[0,3]
        slope = (float)(y2-y1)/(float)(x2-x1)
        if abs(slope) < 0.5:
            continue
        if slope <= 0:
            left_x.append([x1, x2])
            left_y.append([y1, y2])
        else:
            right_x.append([x1, x2])
            right_y.append([y1, y2])
    return left_x, left_y, right_x, right_y


def meanPoint(x):
    sum1 = 0
    sum2 = 0
    for x1, x2 in x:
        sum1 += x1
        sum2 += x2
    sum1 = int(float(sum1)/float(len(x)))
    sum2 = int(float(sum2)/float(len(x)))
    return [sum1, sum2]

def medianPoint(x):
    xx = sorted(x)
    return xx[(int)(len(xx)/2)]
    

def interpolate(list_x, list_y, y):
    x1 = list_x[0]
    x2 = list_x[1]
    y1 = list_y[0]
    y2 = list_y[1]
    return int(float(y - y1) * float(x2-x1) / float(y2-y1) + x1)

    
def lineFitting(image, left_x, left_y, right_x, right_y):
    result = imageCopy(image)
    height = image.shape[0]
    lx = meanPoint(left_x)
    ly = meanPoint(left_y)
    rx = meanPoint(right_x)
    ry = meanPoint(right_y)
    min_y = int(height*0.6)
    max_y = height
    min_x_left = interpolate(lx, ly, min_y)
    max_x_left = interpolate(lx, ly, max_y)
    min_x_right = interpolate(rx, ry, min_y)
    max_x_right = interpolate(rx, ry, max_y)
    cv2.line(result, (min_x_left, min_y), (max_x_left, max_y), (0, 0, 255), 3)
    cv2.line(result, (min_x_right, min_y), (max_x_right, max_y), (0, 0, 255), 3)
    return result

def lane_detection_function(image):
	result = imageCopy(image)
	height, width = result.shape[:2]
	pt1 = (width*0.45, height*0.6)
	pt2 = (width*0.55, height*0.6)
	pt3 = (width*0.95, height*1.0)
	pt4 = (width*0.05, height*1.0)
	roi_corners = np.array([[pt1, pt2, pt3, pt4]], dtype=np.int32)
	result = polyROI(result, roi_corners)

	lower_white_hls = np.array([0, 200, 0])
	upper_white_hls = np.array([179, 255, 255])
	lower_yellow_hls = np.array([15, 30, 115])
	upper_yellow_hls = np.array([35, 200, 255])
	W_hls = splitColor(result, lower_white_hls, upper_white_hls, cv2.COLOR_BGR2HLS)
	Y_hls = splitColor(result, lower_yellow_hls, upper_yellow_hls, cv2.COLOR_BGR2HLS)
	result = W_hls+Y_hls
	result = cannyEdge(result, 100, 200)
	result = houghLinesP(result, 1, np.pi/180, 10, 10)
	lx, ly, rx, ry = splitLines(result)
	result = lineFitting(image, lx, ly, rx, ry)
	return result
	
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


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3-tiny.cfg"
    weightPath = "./bin/yolov3-tiny.weights"
    metaPath = "./cfg/coco.data"
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
    cap = cv2.VideoCapture("../OpenCV_in_Ubuntu/Data/Lane_Detection_Videos/solidWhiteRight.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, frame_resized, thresh=0.05)
        frame_resized = lane_detection_function(frame_resized)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()