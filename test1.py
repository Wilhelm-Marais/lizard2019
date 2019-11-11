
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


img = cv2.imread('objectivePlot9.jpg')

armRadius = 40

#pose transformation from world to camera
rx = np.radians(90)
ry = np.radians(180)
rz = np.radians(90)
cx, sx = np.cos(rx), np.sin(rx)
cy, sy = np.cos(ry), np.sin(ry)
cz, sz = np.cos(rz), np.sin(rz)
Rx = np.array(((1,0,0),(0,cx,-sx),(0,sx,cx)))
Ry = np.array(((cy,0,sy),(0,1,0),(-sy,0,cy)))
Rz = np.array(((cz,-sz,0),(sz,cz,0),(0,0,1)))
R = Rz @ Ry @ Rx
t = np.array(((10),(10),(10)))
#camera matrix
fx,fy = img.shape[1]/4.0,img.shape[0]/4.0
cx,cy = img.shape[1]/2.0,img.shape[0]/2.0
A = np.array(((fx,0,cx),(0,fy,cy),(0,0,1)))

#original points
pointsWorldFrame = np.array(((150,200,150),(-100,20,100),(50,100,10)))
pointCameraFrame = np.transpose(R) @ (pointsWorldFrame - t)
dists = np.array((pointCameraFrame[-1]))
pointCameraFrame /= dists
pointImage = np.round(A @ pointCameraFrame)

for i in range(np.size(pointsWorldFrame,1)-1):
    x1 = pointImage[0:2,i]
    x2 = pointImage[0:2,i+1]
    angle = math.atan2(x2[1]-x1[1],x2[0]-x1[0])
    offset = np.pi/2
    p1 = (x1 + A[0:2,0:2] @ np.array([np.cos(angle - offset), np.sin(angle - offset)]) * armRadius/dists[i]).astype(int)
    p2 = (x1 + A[0:2,0:2] @ np.array([np.cos(angle + offset), np.sin(angle + offset)]) * armRadius/dists[i]).astype(int)
    p3 = (x2 + A[0:2,0:2] @ np.array([np.cos(angle + offset), np.sin(angle + offset)]) * armRadius/dists[i+1]).astype(int)
    p4 = (x2 + A[0:2,0:2] @ np.array([np.cos(angle - offset), np.sin(angle - offset)]) * armRadius/dists[i+1]).astype(int)

    cv2.fillConvexPoly(img,np.array(((p1),(p2),(p3),(p4))),[0,0,255])

cv2.imshow('image',img)
cv2.waitKey(0)