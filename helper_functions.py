#!/usr/bin/python
import cv2
import numpy as np

PREFIX = "WrightStainImages/" 
SUFFIXES = [".jpg", ".png", ".jpeg"]

#=================== IMAGE DATA FUNCTIONS ==========================#
"""
functions that pull data from an image without changing the image itself
"""

def getROI(image, x1, x2, y1, y2):
    return image[x1:x2, y1:y2]


#================== HSV FUNCTIONS ==================================#
def getHSVValues(img):
    image_hsv_values = []
    for x in range(0, img.shape[0]): 
        for y in range(0, img.shape[1]):
            image_hsv_values.append(img[x,y])
    return image_hsv_values

def calculateHSVBoundsAverage(img, margin):
    hsv_values = getHSVValues(img)
    averages = [sum(y)/len(y) for y in zip(*hsv_values)]
    print "AVERAGE HSV: ",str(averages)
    lower = (averages[0]-margin, 100, 100)
    upper = (averages[0]+margin, 100, 100)
    return lower, upper 

def calculateHSVBoundsMode(img, margin, avg_length):
    hsv_values = getHSVValues(img)
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
    modehsv = max(averages)
    print "MODE HUE: ", str(modehsv)
    lower = (modehsv-margin, 100, 100)
    upper = (modehsv+margin, 255, 255)
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


def showImage(windowName, image, key):
    while cv2.waitKey(key):
    	cv2.imshow(windowName, image)

