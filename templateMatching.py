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

w, h = test_img.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = mosaic.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, test_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()


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