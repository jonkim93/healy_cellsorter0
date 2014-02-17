#!/usr/bin/python

import util
from imh import *
from mira import *

LEGAL_LABELS = ("lymphocyte", "other")
INPUT_TO_LABEL = {"y": "lymphocyte",\
				  "n": "other"}


def featureExtractor(roi): # WHAT SHOULD THE ROI INPUT BE??? an hsv image . . .
	features = util.Counter()

	gray = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,lower,upper,cv2.THRESH_BINARY) 

	contours, numContours, hierarchy = getContours(thresh)

	roi_hsv_values = getHSVValues(roi)
	average_hsv = [sum(y)/len(y) for y in zip(*roi_hsv_values)]

	features["contour area"] = float(cv2.contourArea(contours[0]))   # area of first contour
	features["average h"]    = float(average_hsv[0])                 # avg hue
	features["average s"]    = float(average_hsv[1])                 # avg saturation
	features["average v"]    = float(average_hsv[2])                 # avg value

	return features 




def makeTrainingData(roi_list):
	trainingLabels, trainingData = [], [] 
	for roi in roi_list:
		user_input == None
		cv2.imshow("cell", roi)
        cv2.waitKey(33)
        while user_input == None:
	        user_input = raw_input("\toption (y or n): ")
	        if user_input not in ["y","n"]:
	        	print "Not a valid option, please enter y or n "
	        	user_input = None
		features = featureExtractor(roi)
		trainingLabels.append(INPUT_TO_LABEL[user_input])
		trainingData.append(features)
	return trainingData, trainingLabels



