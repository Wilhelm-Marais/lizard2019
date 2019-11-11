# import cv2
# import numpy as np
#
# img = cv2.imread('C:\Users\wmar5627\Desktop\Captures\Capture0.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# #cv2.imshow('image',img)
# #cv2.waitKey()
#
# orb = cv2.ORB()
# # find the keypoints with ORB
# kp = orb.detect(img,None)
# kp, des = orb.compute(gray, kp)
#
# img2 = cv2.drawKeypoints(gray,kp,color=(0,255,0), flags=0)
#
# cv2.imwrite('orb1.png',img)
# print("done")

import numpy as np
import cv2
from matplotlib import pyplot as plt

mosaic_big = cv2.imread(r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\renav20181125_0000\mesh20181125_2213\mosaic\r20181124_014854_lizard_d2_167_horseshoe_circle01.tif',0)
test_img = cv2.imread(r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\i20181124_014854_cv\PR_20181124_014856_733_RM16.png',0)

test_sizex = test_img.shape[0]
test_sizey = test_img.shape[1]
mosaic_size = mosaic_big.shape[0]
pos_est = [int(mosaic_size/2),int(mosaic_size/2)]
mosaic = mosaic_big[pos_est[0]-test_sizex:pos_est[0]+test_sizex,pos_est[1]-test_sizey:pos_est[1]+test_sizey]

# # Initiate STAR detector
#orb = cv2.ORB_create()
orb1 = cv2.ORB_create(nfeatures = 100)
kp1, des1 = orb1.detectAndCompute(test_img,None)

orb2 = cv2.ORB_create(nfeatures = 400)
kp2, des2 = orb2.detectAndCompute(mosaic,None)



# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(test_img,kp1,mosaic,kp2,matches[:20],None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()


# img2 = test_img
# # draw only keypoints location,not size and orientation
# RGB_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# cv2.drawKeypoints(test_img,kp1,RGB_img2,color=(0,255,0), flags=0)
# plt.imshow(RGB_img2),plt.show()

# img2 = mosaic
# # draw only keypoints location,not size and orientation
# RGB_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# cv2.drawKeypoints(mosaic,kp2,RGB_img2,color=(0,255,0), flags=0)
# plt.imshow(RGB_img2),plt.show()

# cv2.imwrite('mosaicFeatures.jpg',RGB_img2)