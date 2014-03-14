#!/usr/bin/python

import util
from imh import *
from mira import *

LEGAL_LABELS = ("lymphocyte", "other")
INPUT_TO_LABEL = {"y": "lymphocyte",\
				  "n": "other"}
FEATURES = ["contour area",\
			"average h",\
			"average s",\
			"average v"]


def featureExtractor(roi, lower=130, upper=255): # WHAT SHOULD THE ROI INPUT BE??? an hsv image . . .
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


def makeTrainingData(roi_list, display_list):
	trainingLabels, trainingData = [], [] 
	for x in xrange(len(roi_list)):
		print "cell ",x
		roi = roi_list[x]
		print "\tlower, upper: "+str(calculateHSVBoundsMode(roi)[0])+\
			", "+str(calculateHSVBoundsMode(roi)[1])
		cv2.imshow("cell", display_list[x])
		cv2.waitKey(33)
		user_input = raw_input("\toption (y or n): ")
		features = featureExtractor(roi)
		trainingLabels.append(INPUT_TO_LABEL[user_input])
		trainingData.append(features)
	print trainingData
	print trainingLabels
	return trainingData,trainingLabels
