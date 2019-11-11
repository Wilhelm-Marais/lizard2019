import numpy as np
import cv2

cap = cv2.VideoCapture(0)
DIM=(640, 480)
K=np.array([[315.6253438915705, 0.0, 338.8106336101351], [0.0, 315.295355656459, 215.66194384464077], [0.0, 0.0, 1.0]])
D=np.array([[-0.033232131957853746], [0.06567426536603721], [-0.10434173026578585], [0.051118686112310935]])
i = 0

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

while 1:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted = cv2.fisheye.undistortImage(gray, K, D)
    undistorted = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


    # Display the resulting frame
    cv2.imshow('frame',undistorted)
    keypress = cv2.waitKey(5)
    if keypress & 0xFF == ord('q'):
        break
    if keypress & 0xFF == ord('s'):
        cv2.imwrite("C:/Users/wmar5627/Desktop/Captures/Capture%d.jpg" %i, gray);
        i = i + 1
        print("saving")
        pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()