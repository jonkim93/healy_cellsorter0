#!/usr/bin/python
import cv2
import numpy as np
import time
import math
from collections import Counter

DEBUG = False
PREFIXES = ["CellBoundImages/", "WrightStainImages/", "CellScope/" ]
SUFFIXES = [".jpg", ".png", ".jpeg", ".tif"]

#=================== IMAGE DATA FUNCTIONS ==========================#
"""
functions that pull data from an image without changing the image itself
"""

def resizeImg(img, scale_factor):
    return cv2.resize(img, (int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)))

def getROI(image, x1, x2, y1, y2):
    return image[x1:x2, y1:y2]

def subDivideImage(img, w_div=4, h_div=4):
    width = img.shape[0]
    height = img.shape[1]
    sub_w = float(width)/float(w_div)
    sub_h = float(height)/float(h_div)
    subdividedimgs = []
    for x in xrange(w_div):
        for y in xrange(h_div):
            #print "x: ",str(sub_w*x)
            #print "y: ",str(sub_h*y)
            subdividedimgs.append(getROI(img, sub_w*x, sub_w*x+sub_w, sub_h*y, sub_h*y+sub_h))
    return subdividedimgs
    


#TODO: ignore all bounding boxes that are inside of each other
def segmentCellsGray(image, mincellsize=10, lower=130, upper=255):
    boundingBoxes = []
    image = blur(image, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,lower,upper,cv2.THRESH_BINARY) 
    
    cv2.imshow("thresh", thresh)
    
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > mincellsize:
            boundingBoxes.append(cv2.boundingRect(contour))
    boxImg = drawBoundingBoxes(image, boundingBoxes)
    return boundingBoxes, boxImg

def segmentCellsCanny(image, mincellsize=10, lower=130, upper=255):

    boundingBoxes = []
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    gray = blur(gray, 7)
    gray = erodeAndDilate(gray, 15, 3)
    cv2.imshow("gray", gray)
    
    canny = cv2.Canny(gray, 5, 50)  # PLAY AROUND WITH THESE VALUES


    # PLAY AROUND WITH THIS STUFF ==================================
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    canny = cv2.dilate(canny, element)
    element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    canny = cv2.erode(canny, element1)
    # ==============================================================
    #circles = getCircles(gray)
    #circleImg = drawCircles(image, circles)
    #cv2.imshow("circles", circleImg)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(gray,lower,upper,cv2.THRESH_BINARY) 
    
    cv2.imshow("canny", canny)
    
    contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print type(contours)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > mincellsize:
            boundingBoxes.append(cv2.boundingRect(contour))
    boxImg = drawBoundingBoxes(image, boundingBoxes)
    return boundingBoxes, boxImg, canny 


def segmentBeadsCanny(image, mincellsize=10, lower=130, upper=255):

    boundingBoxes = []
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    gray = blur(gray, 7)
    #gray = erodeAndDilate(gray, 15, 3)
    cv2.imshow("gray", gray)
    
    canny = cv2.Canny(gray, 5, 50)  # PLAY AROUND WITH THESE VALUES


    # PLAY AROUND WITH THIS STUFF ==================================
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    canny = cv2.dilate(canny, element)
    element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    canny = cv2.erode(canny, element1)
    # ==============================================================
    #circles = cv2.HoughCircles(canny.copy(), cv2.cv.CV_HOUGH_GRADIENT, 2, 10, np.array([]), 40, 60, 5, 1000)
    circles = cv2.HoughCircles(canny.copy(),
                               cv2.cv.CV_HOUGH_GRADIENT,
                               1,
                               50,           # min distance allowed between circles
                               param1=50, 
                               param2=10,    # if too many circles, increase, and vice versa
                               minRadius=5,  # duh
                               maxRadius=20) # duh
    if circles != None:
        print ("NUMBER OF CIRCLES: "+str(len(circles[0])))
    circleImg = drawCircles(image.copy(), circles)
    
    contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print type(contours)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > mincellsize:
            boundingBoxes.append(cv2.boundingRect(contour))
    boxImg = drawBoundingBoxes(image, boundingBoxes)
    return boundingBoxes, boxImg, canny, circleImg, circles  

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
    upper = (averages[0]+margin, 255, 255)
    if DEBUG:
        print "TIME OF EXECUTION", time.time() - start_time, "seconds"
        print "AVERAGE HSV: ",str(averages)
    print "\t\t",str(averages)
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
    if len(hueCounter) != 0:
        modehue = hueCounter.most_common(1)[0][0]
    else:
        #print "no values"
        return None, None, None 

    if DEBUG:
        print "MOST COMMON: ", hueCounter.most_common(1)
        print "TIME OF EXECUTION", time.time() - start_time, "seconds"
        print "MODE HUE: ", str(modehue)

    if modehue > margin and modehue < (255-margin):
        lower = (modehue-margin, 100, 100)
        upper = (modehue+margin, 255, 255)
    elif modehue <= margin:
        lower = (0, 100, 100)
        upper = (modehue+margin, 255, 255)
    elif modehue >= (255-margin):
        lower = (modehue-margin, 100, 100)
        upper = (255, 255, 255)
    return lower, upper, modehue

def getContours(img, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE):
    contours, hierarchy = cv2.findContours(img, mode, method)
    return contours, len(contours), hierarchy

def filterBeads(img, circles, lower, upper):
    circle_roi_list = []
    filtered_beads = []
    #print img.shape
    if circles != None:
        for circle in circles[0]:
            y1 = int(circle[0]-circle[2])
            y2 = int(circle[0]+circle[2])
            x1 = int(circle[1]-circle[2])
            x2 = int(circle[1]+circle[2])
            #print circle 
            #print str(x1)+", "+str(x2)+ ": "+str(y1)+", "+str(y2)
            circle_roi_list.append(getROI(img, x1, x2, y1, y2))
        for x in xrange(len(circle_roi_list)):
            #showImage(circle_roi_list[x])
            hue_mode = calculateHSVBoundsMode(circle_roi_list[x])[2]
            if hue_mode != None:
                #print "HUE_MODE: "+str(hue_mode)
                if hue_mode < upper and hue_mode > lower:
                    filtered_beads.append(circles[0][x])
    return filtered_beads

def distance(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def filterContoursByArea(img, contours, area_lower_threshold=10000, area_upper_threshold=1000000, draw=False):
    areas = []
    filtered = []
    filtered_bounding_boxes = []
    num_big_contours = 0
    for i in xrange(len(contours)):
        contour_area = cv2.contourArea(contours[i])
        areas.append(contour_area)
        if contour_area > area_lower_threshold and contour_area < area_upper_threshold:
            num_big_contours += 1
            filtered.append(contours[i])
            x, y, w, h = cv2.boundingRect(contours[i])
            filtered_bounding_boxes.append((x,y,w,h))
            if draw:
                cv2.drawContours(img, contours, i, (0, 255, 0))
                cv2.rectangle(img,(x,y),(x+w,y+h),(60,140,40),2)
    return num_big_contours, areas, img, filtered, filtered_bounding_boxes

def drawBoundingBoxes(img, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img 

def drawBoundingBox(img, box):
    x, y, w, h = box
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img 

def getCircles(img,minRadius=1, maxRadius=30,method=cv2.cv.CV_HOUGH_GRADIENT):
    img = cv2.Canny(img, 10, 80)
    cv2.imshow("canny", img)
    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 2, 10, np.array([]), 40, 60, 5, 1000)
    #circles = cv2.HoughCircles(img,method,1,20,50,100,minRadius,maxRadius)
    if circles != None:
        print "NUMBER OF CIRCLES: ",str(len(circles[0]))
    return circles 

def drawCircles(img, circles):
    if circles != None:
        circles = np.uint16(np.around(circles))
        for circle in circles:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    else:
        print "ERROR: NO CIRCLES"
    return img 


#=================== IMAGE EDITING FUNCTIONS =======================#

def generalProcess(img, bt_blur_ksize, bt_ed_iter, bt_ed_ksize, thresh_style, upper, lower, at_blur_ksize, at_ed_iter, at_ed_ksize):
    bt_blurred = None
    bt_ed = None
    cvted_img = None
    thresh = None
    at_blurred = None
    at_ed = None
    final = None 

    # bt = before thresholding
    if bt_blur_ksize > 0:
        bt_blurred = blur(img.copy(), bt_blur_ksize)
    if bt_ed_iter > 0:
        if bt_blurred != None:
            bt_ed = erodeAndDilate(bt_blurred.copy(), bt_ed_iter, bt_ed_ksize)
        else:
            bt_ed = erodeAndDilate(img.copy(), bt_ed_iter, bt_ed_ksize)

    if bt_ed != None:
        bt_intermediate = bt_ed
    elif bt_blurred != None:
        bt_intermediate = bt_blurred
    else:
        bt_intermediate = img.copy() 

    #cv2.imshow("bt_intermediate", bt_intermediate)
    # what style of thresholding?
    if thresh_style == "hsv":
        cvted_img = cv2.cvtColor(bt_intermediate.copy(), cv2.COLOR_BGR2HSV)
        thresh = thresholdHSV(cvted_img.copy(), lower, upper)
    elif thresh_style == "gray":
        cvted_img = cv2.cvtColor(bt_intermediate.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(cvted_img.copy(), lower, upper, cv2.THRESH_BINARY)

    # at = after thresholding
    if at_blur_ksize > 0:
        at_blurred = blur(thresh.copy(), at_blur_ksize)
    if at_ed_iter > 0:
        if at_blurred != None:
            at_ed = erodeAndDilate(at_blurred.copy(), at_ed_iter, at_ed_ksize)
        else:
            at_ed = erodeAndDilate(thresh.copy(), at_ed_iter, at_ed_ksize)

    if at_ed != None:
        final = at_ed
    elif at_blurred != None:
        final = at_blurred
    elif thresh != None:
        final = thresh
    elif cvted_img != None:
        final = cvted_img
    elif bt_ed != None:
        final = bt_ed
    elif bt_blurred != None:
        final = bt_blurred
    else:
        final = img
    return (final, bt_blurred, bt_ed, cvted_img, thresh, at_blurred, at_ed) 


"""
functions that actually act on the image and change its state
"""
def thresholdHSV(img, lowerHSV, upperHSV):
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
    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            path = prefix+inputfile+suffix
            img = cv2.imread(path)
            if img != None:
                break
        if img != None:
            break
    if img == None:
        print "ERROR: image not found ", path
        return
    return img 


def showImage(image, windowName="img", key=1):
    while cv2.waitKey(key):
    	cv2.imshow(windowName, image)

