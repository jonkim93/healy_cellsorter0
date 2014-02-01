#!/usr/bin/python

import cv2
import numpy as np 
from helper_functions import *

# NOTE: think about eroding and dilating 
# NOTE: think about adaptive thresholding

DEBUG = True
BLUR = True
ERODE_DILATE = True
LOWER_HSV = (125,15,100)
UPPER_HSV = (165,255,255)

#==================  MAIN  =================================#
"""
based on hsv values 
1) convert to hsv
2) blur
3) threshold
4) erode/dilate 
"""
def main():    
    prefix = "WrightStainImages/" 
    name = "WS2"
    suffix = ".jpeg"

    #name = "WS3"
    #suffix = ".jpg"

    #name = "WS6"
    #suffix = ".png"
    path = name+suffix
    img = loadImage(path)
    original = img.copy()
    cv2.imshow("original", original)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #1

    """
    #NOTE: this will be important later
    hsv_values = getHSVValues(img)
    averages = [sum(y)/len(y) for y in zip(*hsv_values)]
    print str(averages)
    """

    img = blur(img, 11) #2 blur the colored image #ksize 
    img = erodeAndDilate(img, 10, 4) #4 erode/dilate the colored image #iterations + ksize

    thresh                                = thresholdHSV(img.copy(), LOWER_HSV, UPPER_HSV)  #3 threshold the colored image
    smoothed                              = erodeAndDilate(thresh.copy(), 12, 2)  #4 erode/dilate the thresholded image
    contours, tot_num_contours, hierarchy = getContours(smoothed)
    num_big_contours, areas, original     = filterContoursByArea(original, contours, area_threshold=10000, draw=True)
    
    print num_big_contours
    cv2.imshow("thresh", thresh)
    cv2.imshow("smoothed", smoothed)
    cv2.imshow("contours", original)

    #cv2.imwrite(name+"_blurred.png", img)
    cv2.imwrite(name+"_thresh.png", thresh)
    cv2.imwrite(name+"_smoothed.png", smoothed)
    cv2.imwrite(name+"_contours.png", original)

    showImage("contours", original, 1)
    


"""
attempt at grayscaling then thresholding
"""
def gray():
    path = "WS3.jpg"
    #path = "WS6.png"
    img = loadImage(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    cv2.imshow("gray", gray)
    ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY) 
    cv2.imshow("thresh", thresh)
    blurred = blur(thresh, 3)
    """smoothed                              = erodeAndDilate(blurred.copy(), 5, 2)  #4 erode/dilate the thresholded image
    print type(smoothed)
    bgr_image = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
    print type(bgr_image)
    contours, tot_num_contours, hierarchy = getContours(bgr_image)
    num_big_contours, areas, original     = filterContoursByArea(img, contours, area_threshold=100, draw=True)
    print num_big_contours"""
    #img = erodeAndDilate(img)
    showImage("blurred", blurred, 1)



if __name__=='__main__':
	#main()
    gray()



