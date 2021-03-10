import numpy as np
import cv2
import math

#Grid size
grid = (14,9)#corners*corners
#Side of each square in real units
sqsize = 1.5#cm

#Code modified from
#https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
cap = cv2.VideoCapture(0)

cv2.namedWindow("img")

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((grid[0]*grid[1],3), np.float32)
objp[:,:2] = np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)*sqsize
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

calibrated = 0
ret, mtx, dist, rvecs, tvecs=(None,None,None,None,None)
i=0

while 1:
    ret, img = cap.read()
    if calibrated<10:
        i+=1
        if i%20==0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (14,9), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (14,9), corners2, ret)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                calibrated+=1
            cv2.imshow('img', img)
    else:
        #After calibrating with 10 sets of points, save the resulting matrix
        with open("inverseIntrinsicMatrix.np",'wb') as f:
            im = np.linalg.inv(mtx)
            np.save(f,im)
            print(im)
        break
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()