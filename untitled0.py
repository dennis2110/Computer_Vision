# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:38:58 2018

@author: 606C
"""

import numpy as np
import cv2

def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])

def sift_detect(img1, img2, detector='surf'):
    if detector.startswith('si'):
        print ("sift detector......")
        sift = cv2.xfeatures2d.SURF_create()
    else:
        print ("surf detector......")
        sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return bgr_rgb(img3)


img = cv2.imread('Lenna.jpg')
logo_rows, logo_cols = img.shape[:2]

cv2.imshow('src',img)

similarity_mat = np.array([[np.cos(45)/2,-np.sin(45)/2,100],[np.sin(45)/2,np.cos(45)/2,30]])
print(similarity_mat)
#dstimg2 = cv2.warpPerspective(img, similarity_mat, (500,500))
dstimg2 = cv2.warpAffine(img,similarity_mat,(200,200))
cv2.imshow('dst2',dstimg2)


'''
SURF dectect
'''
dectect = sift_detect(img, dstimg2)
cv2.imshow('dectect', dectect)

cv2.waitKey(0)
#cv2.imwrite('dstimg.jpg',dstimg2)
cv2.destroyAllWindows()
