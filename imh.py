#!/usr/bin/python
import cv2
import numpy as np
import time
from collections import Counter

DEBUG = True
PREFIX = "WrightStainImages/" 
SUFFIXES = [".jpg", ".png", ".jpeg"]

#=================== IMAGE DATA FUNCTIONS ==========================#
"""
functions that pull data from an image without changing the image itself
"""

def getROI(image, x1, x2, y1, y2):
    return image[x1:x2, y1:y2]

def segmentCells(image, mincellsize=100):
    boundingBoxes = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        boundingBoxes.append(cv2.boundingRect(contour))
    return boundingBoxes


#================== HSV FUNCTIONS ==================================#
def getHSVValues(img):
    image_hsv_values = []
    for x in range(0, img.shape[0]): 
        for y in range(0, img.shape[1]):
            image_hsv_values.append(img[x,y])
    return image_hsv_values

def calculateHSVBoundsAverage(img, margin=30):
    if DEBUG:
        start_time = time.time()
        print "IMG SHAPE: ",img.shape
    hsv_values = getHSVValues(img)
    averages = [sum(y)/len(y) for y in zip(*hsv_values)]
    lower = (averages[0]-margin, 100, 100)
    upper = (averages[0]+margin, 100, 100)
    if DEBUG:
        print "TIME OF EXECUTION", time.time() - start_time, "seconds"
        print "AVERAGE HSV: ",str(averages)
    return lower, upper 

def calculateHSVBoundsMode(img, avg_length=20, margin=30):
    if DEBUG:
        start_time = time.time()
        print "IMG SHAPE: ",img.shape

    hsv_values = getHSVValues(img)

    """
    averages = []
    firstsum = 0
    for i in xrange(avg_length):
        firstsum += hsv_values[i][0]
    firstavg = firstsum/avg_length
    averages.append(firstavg)
    prevavg = firstavg
    for i in xrange(avg_length,len(hsv_values)):
        prevavg = prevavg + hsv_values[i][0]/avg_length - hsv_values[i-avg_length][0]/avg_length
        averages.append(prevavg)
    modehue = max(averages)
    """

    hues = [x[0] for x in hsv_values]
    hueCounter = Counter(hues)
    modehue = hueCounter.most_common(1)[0][0]

    if DEBUG:
        print "MOST COMMON: ", hueCounter.most_common(1)
        print "TIME OF EXECUTION", time.time() - start_time, "seconds"
        print "MODE HUE: ", str(modehue)

    if modehue > margin and modehue < (255-margin):
        lower = (modehue-margin, 100, 100)
        upper = (modehue+margin, 255, 255)
    elif modehue < margin:
        lower = (0, 100, 100)
        upper = (modehue+margin, 255, 255)
    elif modehue > (255-margin):
        lower = (modehue-margin, 100, 100)
        upper = (255, 255, 255)
    return lower, upper



def getContours(img, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE):
    contours, hierarchy = cv2.findContours(img, mode, method)
    return contours, len(contours), hierarchy

def filterContoursByArea(img, contours, area_threshold=10000, draw=False):
    areas = []
    num_big_contours = 0
    for i in xrange(len(contours)):
        contour_area = cv2.contourArea(contours[i])
        areas.append(contour_area)
        if contour_area > area_threshold:
            num_big_contours += 1
            if draw:
                cv2.drawContours(img, contours, i, (0, 0, 255))
    return num_big_contours, areas, img

#=================== IMAGE EDITING FUNCTIONS =======================#
"""
functions that actually act on the image and change its state
"""
def thresholdHSV(img, lowerHSV, upperHSV, erode_and_dilate=True):
    """ 
    Thresholds an image for a certain range of hsv values 
    """ 
    threshImg = img.copy()
    threshImg = cv2.inRange(img, lowerHSV, upperHSV, threshImg)
    return threshImg

def blur(img, ksize=15):
    img = cv2.medianBlur(img,ksize)
    return img 

def erodeAndDilate(img, iterations=5, ksize=4):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    for i in xrange(iterations):
        img = cv2.erode(img, element)
        img = cv2.dilate(img, element)
    return img 

#==================  DIRECTORY HELPER FUNCTIONS   ====================#

"""
functions that act on files and load images, etc
"""

def loadImage(inputfile):
    suff_ind = 0
    suffix = SUFFIXES[suff_ind]
    for suffix in SUFFIXES:
        path = PREFIX+inputfile+suffix
        img = cv2.imread(path)
        if img != None:
            break
    if img == None:
        print "ERROR: image not found"
        return
    return img 


def showImage(image, windowName="img", key=1):
    while cv2.waitKey(key):
    	cv2.imshow(windowName, image)

