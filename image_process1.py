#!/usr/bin/python


import sys, getopt 
import cv2
import csv

import numpy as np 
from imh import *
from machinelearning import *
from util import *
from mira import *

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

def train(inputfile, outputoption):
    img = loadImage(inputfile)
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    boundingBoxes, boxImg = segmentCells(img, 50 )

    boxImg = cv2.cvtColor(boxImg, cv2.COLOR_BGR2HSV)
    cv2.imshow("box", boxImg)

    roi_list = []

    for i in xrange(len(boundingBoxes)):
        print i
        box = boundingBoxes[i]
        x, y, w, h = box  

        if w >= imgWidth/2 and h >= imgHeight/2:
            pass
        else:
            roi = getROI(boxImg, y, y+h, x, x+w)
            roi_list.append(roi)

    trainingData, trainingLabels = makeTrainingData(roi_list)
    miraClassifier = MiraClassifier(LEGAL_LABELS, max_iterations=5)
    miraClassifier.train(trainingData, trainingLabels)
    weights = miraClassifier.weights
    weights_data = [(LEGAL_LABELS[0], LEGAL_LABELS[1], LEGAL_LABELS[2])]    
    temp = []
    for label in LEGAL_LABELS:
        temp.append(weights[label])

    name = "weights"
    with open(name+'.csv', 'wb') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for datum in weights_data:
            datawriter.writerow(datum)

            
def miraClassify(inputfile, outputoption):
    name = 'weights'
    weights = []
    with open(name+'.csv', 'rb') as csvfile:
        datareader = cvs.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            weights.append(row)

    img = loadImage(inputfile)
    miraClassifier = MiraClassifier(LEGAL_LABELS, max_iterations=5)
    miraClassifier.weights = weights[1]

    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    boundingBoxes, boxImg = segmentCells(img, 50 )

    boxImg = cv2.cvtColor(boxImg, cv2.COLOR_BGR2HSV)
    cv2.imshow("box", boxImg)

    roi_list = []

    for i in xrange(len(boundingBoxes)):
        print i
        box = boundingBoxes[i]
        x, y, w, h = box  

        if w >= imgWidth/2 and h >= imgHeight/2:
            pass
        else:
            roi = getROI(boxImg, y, y+h, x, x+w)
            roi_list.append(roi)
    guesses = miraClassifier.classify(roi_list)

    numOfCells = 0
    for guess in guesses:
        if guess == "lymphocyte":
            numOfCells += 1

    print "NUMBER OF CELLS: ", numOfCells



def hsv(inputfile, outputoption):    
    img = loadImage(inputfile)

    bt_blur_ksize = 0
    bt_ed_iter = 0
    bt_ed_ksize = 0
    thresh_style = "hsv"
    lower = (70,150,100)
    upper = (150,255,255)
    at_blur_ksize = 3
    at_ed_iter = 5
    at_ed_ksize = 3
    
    (final,\
     bt_blurred,\
     bt_ed,\
     cvted_img,\
     thresh,\
     at_blurred,\
     at_ed) = generalProcess(img, bt_blur_ksize, bt_ed_iter, bt_ed_ksize, thresh_style, upper, lower, at_blur_ksize, at_ed_iter, at_ed_ksize)

    #lower, upper = calculateHSVBoundsMode(img)    <---- THIS IS IMPORTANT; adaptive thresholding
    contours, tot_num_contours, hierarchy = getContours(final.copy())
    num_big_contours, areas, cont_img    = filterContoursByArea(img.copy(), contours, area_lower_threshold=10, area_upper_threshold=1000, draw=True)
    
    outputoption = outputoption.lower()
    if 't' in outputoption or 'thresh' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_thresh.png", cvted_img)
    if 's' in outputoption or 'smoothed' in outputoption or 'smooth' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_smoothed.png", at_ed)
    if 'c' in outputoption or 'contour' in outputoption or 'contours' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_contours.png", cont_img)

    print num_big_contours
    if bt_blurred != None: cv2.imshow("bt_blur", bt_blurred) 
    if bt_ed != None: cv2.imshow("bt_ed", bt_ed)
    if cvted_img != None: cv2.imshow("converted", cvted_img)
    if thresh != None: cv2.imshow("thresholded", thresh)
    if at_ed != None: cv2.imshow("smoothed", at_ed)
    if cont_img != None: cv2.imshow("contours", cont_img)
    showImage(cont_img, "contours",1)
    

"""
attempt at grayscaling then thresholding
"""
def gray(inputfile, outputoption):
    img = loadImage(inputfile)

    s = subDivideImage(img, 10, 10)
    img = s[0]
    img = cv2.resize(img, (1000,1000))
    

    bt_blur_ksize = 0
    bt_ed_iter = 0
    bt_ed_ksize = 0
    thresh_style = "gray"
    upper = 255
    lower = 130
    at_blur_ksize = 0 
    at_ed_iter = 0
    at_ed_ksize = 0
    
    (final,\
     bt_blurred,\
     bt_ed,\
     cvted_img,\
     thresh,\
     at_blurred,\
     at_ed) = generalProcess(img, bt_blur_ksize, bt_ed_iter, bt_ed_ksize, thresh_style, upper, lower, at_blur_ksize, at_ed_iter, at_ed_ksize) 
    
    final = cv2.bitwise_not(final.copy())
    cv2.imshow("final", final)

    c = getCircles(final.copy())
    circleImage = drawCircles(img.copy(), c)

    cv2.imshow("sub", circleImage)

    cont = final.copy()

    contours, tot_num_contours, hierarchy = getContours(cont)
    num_big_contours, areas, cont     = filterContoursByArea(img.copy(), contours, area_lower_threshold=10,area_upper_threshold=1000, draw=True)
    print num_big_contours

    if 't' in outputoption or 'thresh' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_thresh.png", cvted_img)
    if 'c' in outputoption or 'contour' in outputoption or 'contours' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_contours.png", cont)
    showImage(thresh, "thresh", 1)


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
    elif thresholdoption == "t" or thresholdoption == "train":
        train(inputfile, outputoption)
    else:
        print "ERROR: no such threshold option; options are 'g' or 'gray' or 'grey' for grayscale, or 'h' or 'hsv' for hsv"
        sys.exit(2)


if __name__=='__main__':
    main(sys.argv[1:])



