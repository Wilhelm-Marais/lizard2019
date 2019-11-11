
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import csv


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2


def ratioTest(matches):

    good = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good

def homography(good,kp1,kp2,mosaic):

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


        M, mask = cv2.findFundamentalMat(src_pts, dst_pts)

        pts1 = src_pts[mask.ravel()==1]
        pts2 = dst_pts[mask.ravel() == 1]
        # M, mask = cv2.estimateAffine2D(src_pts, dst_pts,method =  cv2.RANSAC,ransacReprojThreshold =  5)
        # M = np.concatenate((M,np.array([[0,0,1]])),axis = 0)
        # M = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine = True)
        # M = np.concatenate((M,np.array([[0,0,1]])),axis = 0)

        #matchesMask = mask.ravel().tolist()

        #h, w = test_img.shape
        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #dst = cv2.perspectiveTransform(pts, M)

        #img2 = cv2.polylines(mosaic, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        M = None
        src_pts = None
        dst_pts = None

    return M,pts1,pts2





image_number = 560

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
test_img = test_img_orig #[600:900,600:900]
#test_img = test_img_orig

test_sizex = test_img_orig.shape[0]
test_sizey = test_img_orig.shape[1]
test_sizex = 1600
test_sizey = 1600

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

good = ratioTest(matches)
H, pts1, pts2 = homography(good, kp1, kp2, mosaic)

M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
h, w = test_img.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
img2 = cv2.polylines(mosaic, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, H)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(test_img, mosaic, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 1, H)
lines2 = lines2.reshape(-1, 3)
# img3, img4 = drawlines(mosaic, test_img, lines2, src_pts, dst_pts)
# img5, img6 = drawlines(test_img, mosaic, lines1, src_pts, dst_pts)
img3, img4 = drawlines(mosaic, test_img, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()