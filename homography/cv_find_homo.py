# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:01:53 2018

@author: 606C
"""

import cv2
import numpy as np

x1,y1,x2,y2,x3,y3,x4,y4,x11,y11,x21,y21,x31,y31,x41,y41 = 78,253,134,253,134,280,78,280,277,69,479,133,479,242,277,225
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

print(H1)


print(H1.dot([[x1],[y1],[1]]))
print(H1.dot([[x2],[y2],[1]]))
print(H1.dot([[x3],[y3],[1]]))
print(H1.dot([[x4],[y4],[1]]))

x1_,y1_,c1_ = H1.dot([[x1],[y1],[1]])
x2_,y2_,c2_ = H1.dot([[x2],[y2],[1]])
x3_,y3_,c3_ = H1.dot([[x3],[y3],[1]])
x4_,y4_,c4_ = H1.dot([[x4],[y4],[1]])

x1_ /= c1_
x2_ /= c2_
x3_ /= c3_
x4_ /= c4_

y1_ /= c1_
y2_ /= c2_
y3_ /= c3_
y4_ /= c4_

print(x1_, y1_)
print(x2_, y2_)
print(x3_, y3_)
print(x4_, y4_)