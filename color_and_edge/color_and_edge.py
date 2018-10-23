# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:23:56 2018

@author: 606C
"""

import cv2
import numpy as np

dark_hand = cv2.imread('dark_hand_resized.jpg')
bright_hand = cv2.imread('bright_hand_resized.jpg')
#
dark_hsv = cv2.cvtColor(dark_hand,cv2.COLOR_BGR2HSV)
x,y,_=dark_hand.shape
color1 = np.zeros([x,y])
color1[dark_hsv[:,:,0]<20] = 255
color1[dark_hsv[:,:,0]==0] = 0
bright_hsv= cv2.cvtColor(bright_hand,cv2.COLOR_BGR2HSV)
color2 = np.zeros([x,y])
color2[bright_hsv[:,:,0]<20] = 255
color2[bright_hsv[:,:,0]==0] = 0
#
dark_gray = cv2.cvtColor(dark_hand,cv2.COLOR_BGR2GRAY)
kernal = np.array([[1,1,1],[1,-8,1],[1,1,1]])
d_edge_8 = cv2.filter2D(dark_gray, -1, kernal)
#d_edge_8 = cv2.Laplacian(dark_gray,cv2.CV_32F)
#d_edge_8 = cv2.convertScaleAbs(d_edge_8)
d_edge_4 = cv2.Laplacian(dark_gray,cv2.CV_16S,ksize=3)

#d_sharp_8 = dark_gray + d_edge_8 
#d_sharp_4 = cv2.convertScaleAbs(dark_gray - d_edge_4)

#
bright_gray = cv2.cvtColor(bright_hand,cv2.COLOR_BGR2GRAY)
b_edge_8 = cv2.Laplacian(bright_gray,cv2.CV_32F,ksize = 3)
b_edge_8 = cv2.convertScaleAbs(b_edge_8)
b_edge_4 = cv2.Laplacian(bright_gray,cv2.CV_32F)
b_edge_4 = cv2.convertScaleAbs(b_edge_4)
b_sharp_8 = (bright_gray - b_edge_8).astype('uint8') 
b_sharp_4 = (bright_gray - b_edge_4).astype('uint8') 




#cv2.imshow('src1',dark_hand)
#cv2.imshow('src2',bright_hand)
#cv2.imshow('color1',color1)
#cv2.imshow('color2',color2)
cv2.imshow('d_edge_8',d_edge_8)
cv2.imshow('d_edge_4',d_edge_4)
#cv2.imshow('d_sharp_8',d_sharp_8)
#cv2.imshow('d_sharp_4',d_sharp_4)
'''
cv2.imshow('b_edge_8',b_edge_8)
cv2.imshow('b_edge_4',b_edge_4)
cv2.imshow('b_sharp_8',b_sharp_8)
cv2.imshow('b_sharp_4',b_sharp_4)
'''
cv2.imshow('d_gray',dark_gray)
#cv2.imshow('b_gray',bright_gray)


cv2.waitKey(0)
cv2.destroyAllWindows()