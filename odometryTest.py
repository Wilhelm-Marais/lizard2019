
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import csv
import math
from scipy.spatial.transform import Rotation as Rot


def reject_outliers(data, keypoints, descriptor, pts1,  m=2, index=2):
    condition = abs(data[index,:] - np.mean(data[index,:])) < m * np.std(data[index,:])
    return data[:,condition],keypoints[condition],descriptor[condition],pts1[condition]



def mosaicPixel2LatLong(mosaic, scale, offset, pixel):

    return None


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        #color  = tuple([0,255,0])
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
def getEssential(K,good,kp1,kp2,des1,point_positions):

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        src_keypoints = [kp1[m.queryIdx] for m in good]
        src_des = [des1[m.queryIdx] for m in good]
        dst_positions = [point_positions[m.trainIdx] for m in good]

        M, mask = cv2.findEssentialMat(src_pts, dst_pts, K)

        pts1 = src_pts[mask.ravel()==1]
        pts2 = dst_pts[mask.ravel() == 1]

        keypoints1 = np.array(src_keypoints)[mask.ravel()==1]
        descriptor1 = np.array(src_des)[mask.ravel()==1]
        positions2 = np.array(dst_positions)[mask.ravel()==1]

        return M, pts1, pts2, keypoints1, descriptor1, positions2

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        M = None
        pts1 = None
        pts2 = None
        return None

def getFundamental(good,kp1,kp2,des1):

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        src_keypoints = [kp1[m.queryIdx] for m in good]
        src_des = [des1[m.queryIdx] for m in good]

        M, mask = cv2.findFundamentalMat(src_pts, dst_pts)

        pts1 = src_pts[mask.ravel()==1]
        pts2 = dst_pts[mask.ravel() == 1]

        keypoints1 = np.array(src_keypoints)[mask.ravel()==1]
        descriptor1 = np.array(src_des)[mask.ravel()==1]

        return M, pts1, pts2, keypoints1, descriptor1

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        M = None
        pts1 = None
        pts2 = None
        return None



image_number = 10

mosaic_big = cv2.imread(r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\renav20181125_0000\mesh20181125_2213\mosaic\r20181124_014854_lizard_d2_167_horseshoe_circle01.tif',0)

with open('stereo_poses.txt', newline = '') as file:
    data_reader = csv.reader(file, delimiter='\t')
    for data in data_reader:
        if int(data[0]) == image_number:
            coord_est = [float(data[4]), float(data[5])]
            image_file = r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\i20181124_014854_cv\%s' %data[11]


test_img_orig = cv2.imread(image_file,0)
 #[600:900,600:900]
#test_img = test_img_orig
K = np.array([[1728.199492098595,0,693.6730128000369],
              [0,1731.844266140965,556.3529001673897],
              [0,0,1]])
distCoeffs = np.array([0.1707173157090531,0.7809911798704593,-0.00344067943762903,-0.006809862324102093,-0.6903206265783959])
test_img = cv2.undistort(test_img_orig,K,distCoeffs)
#test_img = test_img[100:1100,100:1100]

test_sizex = test_img_orig.shape[0]
test_sizey = test_img_orig.shape[1]
test_sizex = 1600
test_sizey = 1600

mosaic_size = mosaic_big.shape[0]
scale = np.array([1.0/(1275.9-1259.7),1.0/(-474.5+459.3)])
meters_to_pixels = scale*mosaic_size
pixels_to_meters = 1.0/meters_to_pixels

offset = [1275.9,-474.5]
pos_est = [int((offset[0]-coord_est[0])*meters_to_pixels[0]),int((offset[1]-coord_est[1])*meters_to_pixels[1])]
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

#run sift matches
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = ratioTest(matches)
F, pts1, pts2, keypoints1, descriptor1 = getFundamental(good, kp1, kp2, des1)


lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(test_img, mosaic, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(mosaic, test_img, lines2, pts2, pts1)


a1 = lines1[:,0]
b1 = lines1[:,1]
c1 = lines1[:,2]
xy1 = np.linalg.lstsq(np.concatenate([[a1,b1]]).T,-c1,rcond=None)
a2 = lines2[:,0]
b2 = lines2[:,1]
c2 = lines2[:,2]
xy2 = np.linalg.lstsq(np.concatenate([[a2,b2]]).T,-c2,rcond=None)
#find camera centre from xy2[0]

centre1 = xy1[0]
centre2 = xy2[0]

#distance between intersections and corresponding points => triangulate relative to camera

#find camera angle
#(xy1[0] - centre1)
Kinv = np.linalg.inv(K)
r2 = Kinv@np.array([[centre1[0]],[centre1[1]],[1]])
v = r2/np.linalg.norm(r2)
theta2 = -math.asin(v[0])
theta1 = math.asin(v[1]/math.cos(theta2))
rotx = Rot.from_euler('x', theta1, degrees = False)
roty = Rot.from_euler('y', theta2, degrees = False)

trans = centre2*np.abs(pixels_to_meters)


depths = []
points3D = np.zeros((3,len(pts1)))
rz = []
for i in range(len(pts1)):

    r1 = Kinv@np.append(pts1[i], 1)

    cos_angle = np.dot(r1,r2)/(np.linalg.norm(r1)*np.linalg.norm(r2))
    angle = math.acos(cos_angle)

    distance = np.squeeze((pts2[i] - centre2)*np.abs(pixels_to_meters))
    depth = np.linalg.norm(distance)/math.tan(angle)
    depths.append(depth)

    points3D[:,i] = (distance[0],distance[1],depth)
    # vector from camera centre through point in world frame
    v2 = Rot.as_dcm(roty)@Rot.as_dcm(rotx)@r1
    #2d component which has to align with epipolar lines
    #v2 = v2[0:2]/np.linalg.norm(v2[0:2])
    v1 = np.squeeze(pts2[i] - centre2)
    theta3 = math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
    theta3 = (theta3 + 2*math.pi) % (2*math.pi)
    rz.append(theta3)

R1 = Rot.as_dcm(Rot.from_euler('xyz', [theta1,theta2,np.average(rz)], degrees = False))
t1 = np.array([[0],[0],[0]])

for i in range(50):
    des3 = des1
    kp3 = kp1

    #feature positions in world frame
    image_number = image_number + 1
    with open('stereo_poses.txt', newline = '') as file:
        data_reader = csv.reader(file, delimiter='\t')
        for data in data_reader:
            if int(data[0]) == image_number:
                coord_est = [float(data[4]), float(data[5])]
                image_file = r'C:\Users\wmar5627\Desktop\r20181124_014854_lizard_d2_167_horseshoe_circle01\i20181124_014854_cv\%s' %data[11]

    test_img_orig = cv2.imread(image_file,0)
    test_img = cv2.undistort(test_img_orig,K,distCoeffs)
    kp1, des1 = sift.detectAndCompute(test_img, None)


    #find matches between successive frame using features with known depth
    des2 = descriptor1
    kp2 = keypoints1
    matches = flann.knnMatch(des1, des2, k=2)
    good = ratioTest(matches)
    #find pose estimate using known 3d positions
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_positions = [points3D[:,m.trainIdx] for m in good]

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(dst_positions), src_pts, K, distCoeffs=None, rvec = cv2.Rodrigues(R1)[0], tvec= np.array([t1[0],t1[1],t1[2]]), useExtrinsicGuess = True)

    R2 = cv2.Rodrigues(rvec)[0]
    t2 = tvec

    # find all matches between successive frames
    matches = flann.knnMatch(des1, des3, k=2)
    good = ratioTest(matches)

    # pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # pts3 = np.float32([kp3[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #
    F, pts1, pts3, keypoints1, descriptor1 = getFundamental(good, kp1, kp3, des1)

    projMatr1 = K @ np.concatenate((R1, t1), axis=1)
    projMatr2 = K @ np.concatenate((R2, t2), axis=1)
    points4D = cv2.triangulatePoints(projMatr1, projMatr2, pts3, pts1)

    points3D,keypoints1,descriptor1,pts1 = reject_outliers(points4D[0:3,:]/points4D[-1,:],keypoints1,descriptor1,pts1)

    #for triangulated points which were previously used in odometery,
    #filter the values
    for previous3Dpoints in np.array(dst_positions)[inliers]:
        if keypoints1.pt in src_pts[inliers]:
            print("hit")
        pass

    print("i:",i)
    print("trans",t2)

    t1 = t2
    R1 = R2
