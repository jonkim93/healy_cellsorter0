#!/usr/bin/python


import sys, getopt 
import cv2
import csv

import numpy as np 
from imh import *
from machinelearning import *
from util import *
from mira import *

#TODO: ADD IN PARSE OPTION FOR MIRACLASSIFY
#

SCALE_FACTOR = 0.3
DEBUG = True
BLUR = True
ERODE_DILATE = True
LOWER_HSV = (125,15,100)
UPPER_HSV = (165,255,255)

SUBDIVIDE = True
RESIZE = True

PREFIX = "ProcessedImages/"

#==================  MAIN  =================================#
"""
based on hsv values 
1) convert to hsv
2) blur
3) threshold
4) erode/dilate 
"""

def train(img, inputfile, outputoption):
    

    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    CELL_SEG_OPTION = "CANNY" #"GRAY"

    if CELL_SEG_OPTION == "CANNY":
        boundingBoxes, boxImg, canny  = segmentCellsCanny(img, 100, 150, 255 ) #image, mincellsize, lower, upper
    elif CELL_SEG_OPTION == "GRAY":
        boundingBoxes, boxImg = segmentCellsGray(img, 10, 150, 255 ) #image, mincellsize, lower, upper

    hsvBoxImg = cv2.cvtColor(boxImg.copy(), cv2.COLOR_RGB2HSV)

    print "BOX IMG SHAPE: "+str(boxImg.shape)
    if boxImg.shape[1] > 1500 or boxImg.shape[0] > 1500:
        dispBoxImg = resizeImg(boxImg.copy(), SCALE_FACTOR)#float(1500/imgWidth))
    else:
        dispBoxImg = boxImg.copy()
    cv2.imshow("box", dispBoxImg)

    roi_list = []
    display_roi_list = []

    for i in xrange(len(boundingBoxes)):
        #print i
        box = boundingBoxes[i]
        x, y, w, h = box  

        if w >= imgWidth/8 and h >= imgHeight/8:
            pass
        else:
            roi = getROI(hsvBoxImg, y, y+h, x, x+w)
            display_roi_list.append(getROI(boxImg, y, y+h, x, x+w))
            roi_list.append(roi)
    if len(roi_list) > 0:
        trainingData, trainingLabels = makeTrainingData(roi_list, display_roi_list)
        miraClassifier = MiraClassifier(LEGAL_LABELS, max_iterations=5)
        miraClassifier.train(trainingData, trainingLabels)
        weights = miraClassifier.weights
        print weights 
        weights_data = [("FEATURES", LEGAL_LABELS[0], LEGAL_LABELS[1])]    
        for feature in FEATURES:
            weights_data.append((feature, weights[LEGAL_LABELS[0]][feature], \
                weights[LEGAL_LABELS[1]][feature]))
        #print temp
        name = "weights"
        with open(name+'.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for datum in weights_data:
                datawriter.writerow(datum)
    return None 
            
def miraClassify(img, inputfile, outputoption):
    name = 'weights'
    weights = []
    with open(name+'.csv', 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            weights.append(row)

    print weights

    mira_weights = {}
    for l in LEGAL_LABELS:
        mira_weights[l] = util.Counter()
    
    for y in xrange(1, len(weights)):
        for a in xrange(len(LEGAL_LABELS)):
            mira_weights[LEGAL_LABELS[a]][weights[y][0]] = float(weights[y][a+1])

    print mira_weights
    miraClassifier = MiraClassifier(LEGAL_LABELS, max_iterations=5)
    miraClassifier.weights = mira_weights

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
    roi_feature_list = []
    for roi in roi_list:
        roi_feature_list.append(featureExtractor(roi))
    guesses = miraClassifier.classify(roi_feature_list)

    numOfCells = 0
    for guess in guesses:
        if guess == "lymphocyte":
            numOfCells += 1
    print guesses

    print "NUMBER OF CELLS: ", numOfCells
    return None


def hsv(img, inputfile, outputoption):    

    bt_blur_ksize = 0
    bt_ed_iter = 5
    bt_ed_ksize = 3
    thresh_style = "hsv"
    lower = (100,50,50)
    upper = (150,255,255)
    at_blur_ksize = 7
    at_ed_iter = 25
    at_ed_ksize = 7
    
    (final,\
     bt_blurred,\
     bt_ed,\
     cvted_img,\
     thresh,\
     at_blurred,\
     at_ed) = generalProcess(img, bt_blur_ksize, bt_ed_iter, bt_ed_ksize, thresh_style, upper, lower, at_blur_ksize, at_ed_iter, at_ed_ksize)

    #lower, upper = calculateHSVBoundsMode(img)    <---- THIS IS IMPORTANT; adaptive thresholding
    contours, tot_num_contours, hierarchy = getContours(final.copy())
    num_big_contours, areas, cont_img    = filterContoursByArea(img.copy(), contours, area_lower_threshold=1000, area_upper_threshold=100000, draw=True)
    
    outputoption = outputoption.lower()
    if 't' in outputoption or 'thresh' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_thresh.png", cvted_img)
    if 's' in outputoption or 'smoothed' in outputoption or 'smooth' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_smoothed.png", at_ed)
    if 'c' in outputoption or 'contour' in outputoption or 'contours' in outputoption:
        cv2.imwrite(PREFIX+inputfile+"_contours.png", cont_img)

    print num_big_contours
    images = [img, bt_blurred, bt_ed, cvted_img, thresh, at_ed, cont_img]
    images_names = ("original",\
                    "before threshold blur",\
                    "before threshold erode/dilate",\
                    "converted image",\
                    "thresholded image",\
                    "after threshold erode/dilate",\
                    "contour image")
    for x in xrange(0, len(images)):
        if images[x] != None:
            cv2.namedWindow(images_names[x], cv2.cv.CV_WINDOW_NORMAL)
            #images[x] = resizeImg(images[x], SCALE_FACTOR)
            cv2.imshow(images_names[x], images[x])
            cv2.imwrite(inputfile+"_"+ images_names[x] +".png", images[x])
    #showImage(images[-1], "contours",1)
    return None 
    

"""
attempt at grayscaling then thresholding
"""
def gray(img, inputfile, outputoption):

    

    if SUBDIVIDE:
        s = subDivideImage(img, 10, 10)
        img = s[0]
        img = cv2.resize(img, (1000,1000))
    if RESIZE:
        img = resizeImg(img.copy(), SCALE_FACTOR)

    bt_blur_ksize = 0
    bt_ed_iter = 0
    bt_ed_ksize = 0
    thresh_style = "gray"
    upper = 255
    lower = 50
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

    #final = cvted_img.copy()
    final = cv2.bitwise_not(final.copy())
    cv2.imshow("final", final)

    c = getCircles(img.copy())
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
    return None 

def canny(img, inputfile, outputoption):
    
    # modified parameters
    bead_lower_hue =  5 #SUPT1    # CS: 130      # LOOOK HEREEEEEE
    bead_upper_hue =  25 #SUPT1    # CS: 255     # LOOOK HEREEEEEE
    cell_hsv_lower =  (0,50,50)    # CS: (100,50,50)      # CHANGE THIS
    cell_hsv_upper =  (15,255,255)    # CS: (150,255,255)    # CHANGE THIS
    MAX_DISTANCE = 1000


    boundingBoxes, boxImg, canny, circleImage, circles = segmentBeadsCanny(img.copy(), 100, 150, 255)
    filteredBeads = filterBeads(img, circles, bead_lower_hue, bead_upper_hue) #filteredBeads is an array of 3-vectors of form (x,y,radius)
    beadCenters = []
    for bead in filteredBeads:
        beadCenters.append((bead[0], bead[1]))

    bt_blur_ksize = 0
    bt_ed_iter = 5
    bt_ed_ksize = 3
    thresh_style = "hsv"
    at_blur_ksize = 7
    at_ed_iter = 25
    at_ed_ksize = 7
    area_lower_threshold = 1000
    area_upper_threshold = 10000


    (final,\
     bt_blurred,\
     bt_ed,\
     cvted_img,\
     thresh,\
     at_blurred,\
     at_ed) = generalProcess(img, bt_blur_ksize, bt_ed_iter, bt_ed_ksize, thresh_style, cell_hsv_upper, cell_hsv_lower, at_blur_ksize, at_ed_iter, at_ed_ksize)
    
    contours, tot_num_contours, hierarchy = getContours(final.copy())
    num_big_contours, areas, cont_img, filteredCells, filteredBoundingBoxes  = filterContoursByArea(img.copy(), \
        contours, area_lower_threshold, area_upper_threshold, draw=True)
    cellCenters = []
    for cell in filteredBoundingBoxes:
        cellCenters.append((cell[0]+cell[2]/2, cell[1]+cell[3]/2, (cell[2]+cell[3])/2))

    
    beadAttachedCells = [[]]
    for cell in cellCenters:
        for bead in beadCenters:
            if distance((cell[0], cell[1]),bead) < MAX_DISTANCE:
                beadAttachedCells[0].append(cell)
                break
    print "BEAD ATTACHED CELLS"
    print beadAttachedCells
    print "NUMBER OF BEADS: "+str(len(beadCenters))
    finalCellImg = drawCircles(img.copy(), beadAttachedCells)
    print "NUMBER OF CELLS: "+str(len(beadAttachedCells[0])) + "\n"
    cv2.imwrite(PREFIX+inputfile+"_thresh.png", thresh)
    cv2.imwrite(PREFIX+inputfile+"_final.png", finalCellImg)
    cv2.imwrite(PREFIX+inputfile+"_canny.png", canny)
    cv2.imwrite(PREFIX+inputfile+"_boxes.png", boxImg)
    cv2.imwrite(PREFIX+inputfile+"_circles.png", circleImage)
    return len(beadAttachedCells[0])



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
    process_function = None
    if thresholdoption == "g" or thresholdoption == "gray" or thresholdoption == "grey":
        process_function = gray
    elif thresholdoption == "h" or thresholdoption == "hsv":
        process_function = hsv 
    elif thresholdoption == "t" or thresholdoption == "train":
        process_function = train 
    elif thresholdoption == "m" or thresholdoption == "machinelearn":
        process_function = miraClassify
    elif thresholdoption == "c" or thresholdoption == "canny":
        process_function = canny 
    else:
        print "ERROR: no such threshold option; options are 'g' or 'gray' or 'grey' for grayscale, or 'h' or 'hsv' for hsv"
        sys.exit(2)

    img = loadImage(inputfile)

    totalCellCount = 0
    if process_function != None:
        if SUBDIVIDE:
            s = subDivideImage(img, 4, 4)
            for i in xrange(len(s)):
                totalCellCount += process_function(s[i], inputfile+"_"+str(i), outputoption)
            print "TOTAL CELL COUNT: "+str(totalCellCount)
            cv2.waitKey(1)
        else:
            totalCellCount += process_function(img, inputfile, outputoption)


if __name__=='__main__':
    main(sys.argv[1:])



