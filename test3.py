import numpy as np
import cv2

from matplotlib import pyplot as plt


cameraMatrix1=np.array(((4161.221,0,1445.577),(0,4161.221,984.686),(0,0,1)))
cameraMatrix2=np.array(((4161.221,0,1654.636),(0,4161.221,984.686),(0,0,1)))
distCoeffs1 = np.zeros((1,4))
distCoeffs2 = np.zeros((1,4))
doffs=209.059
baseline=176.252
width=2880
height=1988
imageSize = np.array(((width),(height)))
#imageSize =  1000
R = np.identity(3)
T = np.array(((baseline),(0),(0)))
ndisp=280
isint=0
vmin=25
vmax=248
dyavg=0
dymax=0
R1,R2,P1,P2,Q,validPixROI1,validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, tuple(imageSize), R, T)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R,P1, tuple(imageSize), cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R,P2, tuple(imageSize), cv2.CV_32FC1)

imgL = cv2.imread('im0.png',0)
imgR = cv2.imread('im1.png',0)
fixedLeft = cv2.remap(imgL,leftMapX,leftMapY,interpolation=cv2.INTER_LINEAR)
fixedRight = cv2.remap(imgR,rightMapX,rightMapY,interpolation=cv2.INTER_LINEAR)

# imgLg = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
# imgRg = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=16*20, blockSize=15)
disparity = stereo.compute(fixedLeft,fixedRight)
plt.imshow(disparity/2048,'gray')
plt.show()
cv2.waitKey(0)