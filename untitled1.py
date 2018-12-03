# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:46:05 2018

@author: 606C
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('Fig1.jpg')
rows, cols = img.shape[:2]
x1,y1,x2,y2,x3,y3,x4,y4 = 265,40,839,47,1151,383,0,375
x11,y11,x21,y21,x31,y31,x41,y41 = 300,100,800,100,800,400,300,400
p1 = [x1,y1]
p2 = [x2,y2]
p3 = [x3,y3]
p4 = [x4,y4]
p11 = [x11,y11]
p21 = [x21,y21]
p31 = [x31,y31]
p41 = [x41,y41]
src_pts = np.float32(np.array([p1, p2, p3, p4]))
dst_pts = np.float32(np.array([p11, p21, p31, p41]))
H1, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
dstimg = cv2.warpPerspective(img, H1, (1300,500))

cv2.imshow('src', img)
#cv2.imshow('dst', dstimg)
'''
hough line
'''

def line_image(image):
    dark_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    x,y,_=image.shape
    color1 = np.zeros([x,y])
    mask1 = dark_hsv[:,:,1]>150
    mask2 = dark_hsv[:,:,0]>10
    mask5 = mask2 & mask1
    color1[mask5] = 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line", image)

img2 = cv2.imread('Fig1.jpg') 

dark_hsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
x,y,_=img2.shape
color1 = np.zeros([x,y],dtype=np.uint8)
mask1 = dark_hsv[:,:,1]>150
mask2 = dark_hsv[:,:,0]>10
mask5 = mask2 & mask1
color1[mask5] = 255
cv2.imshow('color',color1)
blurr=cv2.blur(color1,(3,3))
kernal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
edges = cv2.filter2D(blurr, cv2.CV_8U, kernal)
#edges = cv2.Canny(color1,100,200)
retval, edges2 = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('edges',edges)
cv2.imshow('edges2',edges2)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
print(lines.shape)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))
    cv2.line(img2, (x1, y1), (x2, y2), (255, 255, 0), 2)
cv2.imshow("line", img2)


#line_image(img2)


cv2.waitKey(0)
#cv2.imwrite('Fig1_dst.jpg',dstimg)
cv2.destroyAllWindows()