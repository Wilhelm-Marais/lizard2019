
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import csv

image_number = 290

mosaic_big = cv2.imread(r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\renav20181125_0000\mesh20181125_2213\mosaic\r20181124_014854_lizard_d2_167_horseshoe_circle01.tif',0)

with open('stereo_poses.txt', newline = '') as file:
    data_reader = csv.reader(file, delimiter='\t')
    for data in data_reader:
        if int(data[0]) == image_number:
            coord_est = [float(data[4]), float(data[5])]
            image_file = r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\i20181124_014854_cv\%s' %data[11]
#
# poses = file.readlines()
# pose = poses[image_number]

#coord_est = [1271.2, -467.0]




test_img_orig = cv2.imread(image_file,0)
test_img = test_img_orig[600:900,600:900]
#test_img = test_img_orig

test_sizex = test_img_orig.shape[0]
test_sizey = test_img_orig.shape[1]
test_sizex = 750
test_sizey = 750

mosaic_size = mosaic_big.shape[0]

scale = [1.0/(1275.9-1259.7),1.0/(-474.5+459.3)]
offset = [1275.9,-474.5]
pos_est = [int((offset[0]-coord_est[0])*scale[0]*mosaic_size),int((offset[1]-coord_est[1])*scale[1]*mosaic_size)]
mosaic = mosaic_big[pos_est[0]-test_sizex:pos_est[0]+test_sizex,pos_est[1]-test_sizey:pos_est[1]+test_sizey]

start_time = time.time()

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(test_img,None)
kp2, des2 = sift.detectAndCompute(mosaic,None)

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)



# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    M, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.RANSAC, 1.0)
    #M, mask = cv2.estimateAffine2D(src_pts, dst_pts,method =  cv2.RANSAC,ransacReprojThreshold =  5)
    #M = np.concatenate((M,np.array([[0,0,1]])),axis = 0)
    #M = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine = True)
    #M = np.concatenate((M,np.array([[0,0,1]])),axis = 0)

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