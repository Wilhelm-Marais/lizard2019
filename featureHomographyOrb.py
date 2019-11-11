
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

mosaic_big = cv2.imread(r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\renav20181125_0000\mesh20181125_2213\mosaic\r20181124_014854_lizard_d2_167_horseshoe_circle01.tif',0)
test_img = cv2.imread(r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\i20181124_014854_cv\PR_20181124_014856_733_RM16.png',0)

test_sizex = test_img.shape[0]
test_sizey = test_img.shape[1]
mosaic_size = mosaic_big.shape[0]
pos_est = [int(mosaic_size/2),int(mosaic_size/2)]
mosaic = mosaic_big[pos_est[0]-test_sizex:pos_est[0]+test_sizex,pos_est[1]-test_sizey:pos_est[1]+test_sizey]

start_time = time.time()

# Initiate SIFT detector
orb = cv2.cv2.ORB_create(nfeatures = 2000)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(test_img,None)
kp2, des2 = orb.detectAndCompute(mosaic,None)

#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# Match descriptors.
#matches = bf.match(des1,des2)
#matches = bf.knnMatch(des1,des2, k = 3)

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2, k = 2)

MIN_MATCH_COUNT = 10

#matches = sorted(matches, key = lambda x:x.distance)
# store all the good matches as per Lowe's ratio test.
# good = []
# count = 0
# for m in matches:
#     count = count + 1
#     if count < 100:
#         good.append(m)

good = []
for i in range(len(matches)):
    if len(matches[i]) == 2:
        if matches[i][0].distance < 0.7*matches[i][1].distance:
            good.append(matches[i][0])


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = test_img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(mosaic,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

print("time",  time.time()-start_time)
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(test_img,kp1,mosaic,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()