#!/usr/bin/python

import imh
import IPython
import cv2
img = imh.loadImage("WS2.1")

img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 

#imh.showImage(img)

l, u = imh.calculateHSVBoundsMode(img)

print "\n\n"
