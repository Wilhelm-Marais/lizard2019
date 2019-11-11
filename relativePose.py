import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import csv



def getFrame(cap,map1,map2):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Display the resulting frame
    #cv2.imshow('frame', undistorted)
    return undistorted

def ratioTest(matches):

    good = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good

def findRelativePose(good,kp1,kp2,K):

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, threshold=1.0)
        retval, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)

        return R,t

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        return None



# When everything done, release the capture

cap = cv2.VideoCapture(0)
DIM = (640, 480)
K = np.array(
    [[315.6253438915705, 0.0, 338.8106336101351], [0.0, 315.295355656459, 215.66194384464077], [0.0, 0.0, 1.0]])
D = np.array([[-0.033232131957853746], [0.06567426536603721], [-0.10434173026578585], [0.051118686112310935]])
i = 0

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

sift = cv2.xfeatures2d.SIFT_create()
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


if __name__ == '__main__':

    #while 1:

    img1 = getFrame(cap,map1,map2)
    kp1, des1 = sift.detectAndCompute(img1, None)
    print("cap1")
    while 1:
        cv2.waitKey(100)

        img2 = getFrame(cap, map1, map2)
        kp2, des2 = sift.detectAndCompute(img2, None)

        matches = flann.knnMatch(des2, des1, k=2)
        good = ratioTest(matches)

        R,t = findRelativePose(good, kp2, kp1, K)

        print("R:", R)
        print("t:",) np.linalg.norm(t)



        # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
        #                    singlePointColor=None,
        #                    matchesMask=matchesMask,  # draw only inliers
        #                    flags=2)
        #
        # img3 = cv2.drawMatches(test_img, kp1, mosaic, kp2, good, None, **draw_params)
        # plt.imshow(img3, 'gray'), plt.show()

