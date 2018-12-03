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

img1 = cv2.imread('img1.jpg',0)
img2 = cv2.imread('img2.jpg',0)
CDG1 = cv2.imread('CDG1.jpg',0)
CDG3 = cv2.imread('CDG3.jpg',0)

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100,nOctaves=4,nOctaveLayers=3,
                                   extended=False,upright=False)
###
surf.setHessianThreshold(10000)
kp1, des1=surf.detectAndCompute(img1,None)
print(len(kp1))
print(des1.shape)
img3 = cv2.drawKeypoints(img1,kp1,None,(0,255,0),4)
###
surf.setHessianThreshold(11000)
kp2, des2=surf.detectAndCompute(img1,None)
print(len(kp2))
print(des2.shape)
img5 = cv2.drawKeypoints(img1,kp2,None,(0,255,0),4)
###
img4 = sift_detect(CDG1, CDG3)


cv2.imshow('456',img3)
cv2.imshow('789',img5)
#cv2.imshow('match',img4)
#cv2.imshow('123',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
