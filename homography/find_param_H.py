# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:30:59 2018

@author: 606C
"""

import cv2
import numpy as np
logoimg = cv2.imread('img2.png')
print(logoimg.shape)
logo_rows, logo_cols = logoimg.shape[:2]
x1,y1,x2,y2,x3,y3,x4,y4,x11,y11,x21,y21,x31,y31,x41,y41 = 0,0,logo_cols,0,logo_cols,logo_rows,0,logo_rows,277,69,479,133,479,242,277,225
W = np.array([[x1,y1,1,0,0,0,-x1*x11,-y1*x11], 
              [0,0,0,x1,y1,1,-x1*y11,-y1*y11],
              [x2,y2,1,0,0,0,-x2*x21,-y2*x21],
              [0,0,0,x2,y2,1,-x2*y21,-y2*y21],
              [x3,y3,1,0,0,0,-x3*x31,-y3*x31],
              [0,0,0,x3,y3,1,-x3*y31,-y3*y31],
              [x4,y4,1,0,0,0,-x4*x41,-y4*x41],
              [0,0,0,x4,y4,1,-x4*y41,-y4*y41]])
X = np.array([[x11],
              [y11],
              [x21],
              [y21],
              [x31],
              [y31],
              [x41],
              [y41]])

H = np.dot(np.dot(np.linalg.inv(W.T.dot(W)),W.T),X)
#print(W)
#print(H)
H = np.append(H, 1)
print(H.shape)
H = H.reshape([3,3])
'''
print(H1.dot([[x1],[y1],[1]]))
print(H1.dot([[x2],[y2],[1]]))
print(H1.dot([[x3],[y3],[1]])) 
print(H1.dot([[x4],[y4],[1]]))
'''
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






img = cv2.imread('img1.jpg')


x = 78
y = 253
w = 134-78
h = 280-253
crop_img = img[y:y+h,x:x+w]
emptyImage = np.zeros(img.shape, np.uint8)
emptyImage[y:y+h,x:x+w] = crop_img


rows,cols = img.shape[:2]
dstimg = cv2.warpPerspective(logoimg, H1, (cols,rows))
grayimg = cv2.cvtColor(dstimg, cv2.COLOR_BGR2GRAY)
ret,threshold = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY_INV)
threshold_inv = cv2.bitwise_not(threshold) 

#img.copyTo()
masked = cv2.bitwise_and(img, img, mask=threshold)

out = cv2.add(masked,dstimg)

cv2.imshow('aaa',img)
cv2.imshow('bbb',masked)
cv2.imshow('ccc',out)
cv2.waitKey(0)
cv2.imwrite('dstimg.jpg',out)
cv2.destroyAllWindows()