# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:30:59 2018

@author: 606C
"""

import cv2
#import matplotlib.pyplot as plt

mode = True

def get_xy_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x= ',x)
        print('y= ',y)

img1 = cv2.imread('Fig1.jpg')

cv2.namedWindow('img1')
cv2.setMouseCallback('img1',get_xy_point)
while(1):
    cv2.imshow('img1',img1)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()