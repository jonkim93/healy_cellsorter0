#!/usr/bin/python


import sys, getopt 
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


def hsv(inputfile, outputoption):    
    img = loadImage(inputfile)

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
    
    #cv2.imwrite(name+"_blurred.png", img)
    outputoption = outputoption.lower()
    if 't' in outputoption or 'thresh' in outputoption:
        cv2.imwrite(inputfile+"_thresh.png", thresh)
    if 's' in outputoption or 'smoothed' in outputoption or 'smooth' in outputoption:
        cv2.imwrite(inputfile+"_smoothed.png", smoothed)
    if 'c' in outputoption or 'contour' in outputoption or 'contours' in outputoption:
        cv2.imwrite(inputfile+"_contours.png", original)

    print num_big_contours
    cv2.imshow("thresh", thresh)
    cv2.imshow("smoothed", smoothed)
    cv2.imshow("contours", original)

    showImage("contours", original, 1)
    

"""
attempt at grayscaling then thresholding
"""
def gray(inputfile, outputoption):
    img = loadImage(inputfile)
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


def main(argv):
    inputfile = ''
    thresholdoption=''
    outputoption = ''
    try:
        opts, args = getopt.getopt(argv,"hi:t:o:",["ifile=","threshopt=","oopt="])
    except getopt.GetoptError:
        print 'image_process1.py -i <inputfile> -t <thresholdoption> -o <outputoption>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -t <thresholdoption> -o <outputoption>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--thresholdoption"):
            thresholdoption = arg 
        elif opt in ("-o", "--oopt"):
            outputoption = arg
    print 'Input file is: ', inputfile
    print 'Threshold option is:', thresholdoption
    print 'Output option is: ', outputoption
    if thresholdoption == "g" or thresholdoption == "gray" or thresholdoption == "grey":
        gray(inputfile, outputoption)
    elif thresholdoption == "h" or thresholdoption == "hsv":
        hsv(inputfile, outputoption)
    else:
        print "ERROR: no such threshold option; options are 'g' or 'gray' or 'grey' for grayscale, or 'h' or 'hsv' for hsv"
        sys.exit(2)


if __name__=='__main__':
    main(sys.argv[1:])



