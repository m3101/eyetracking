"""
    Position estimation demonstration/3d window prototype
    Copyright (C) 2021 Amélia O. F. da S.
    <a.mellifluous.one@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

#Parameters for projection
## Measure your pupil separation in real units
## separation_at_ref can be obtained using the tracker
## Stay at a real reference_distance, track your eyes
## and register the pixel separation (see the top left corner of the
## tracker screen)
eyeseparation=6#cm
reference_distance=40#cm
separation_at_ref=100#pixels
##For projecting
##Measure the projecting window width
## in pixels divided by its real width
## in real units
screen_pixel_density = 640/16 ##pixels/cm

#Parameters for detecting tracking loss
## Series length
eyeSeriesLength = 20
## Sensitivity for tracking loss (how many times the variance of the last
## point has to be bigger than the set variance for us to consider it an outlier)
eyeSeriesSensitivity = 0.40

#Whether the estimation should be made via camera matrices or
#the default proportion method
usematrix = True

#Smoothing factors
smoothingf = 0.6
trackingsmoothfactor = 0.1

#Tracking radii
trackradius = 14
kernelsize = 3

import cv2
import numpy as np
import math
import utils
import pdb
import faceCascade

#If we're using the intrinsic matrix, we have to load it
if usematrix:
    try:
        with open("inverseIntrinsicMatrix.np",'rb') as f:
            iimatrix = np.load(f)
    except:
        print("Calibration matrix not found.")
        print("Please change the method flag on the script or calibrate the camera with calibrate.py")
        exit(1)

#Pupil positions (and kernels/images)
corners = np.array([[0,0],[0,0]])
cimgs = [None,None]

#Smoothed error
trackerror = np.zeros(1)

#A series of the last N eye positions so we can detect when it loses track
eyeSeries = None

#Windows
name = "Tracker"
cv2.namedWindow(name,cv2.WINDOW_NORMAL)

name2 = "Projection"
cv2.namedWindow(name2,cv2.WINDOW_NORMAL)

name3 = "3d static image"
cv2.namedWindow(name3,cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)

#Our images to project
llyn = cv2.imread("llyn.jpg")

#Final coordinates
pos_3d = np.zeros(3)

#Main loop
while(True):
    #Caputre and process the frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    #frame = cv2.flip(frame,1)
    #Grayscale and median blur for lessening noise
    frameg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frameg = cv2.medianBlur(frameg,7)

    #If we don't have a tracking series,
    #find a face and track the eyes
    if eyeSeries is None:
        eyes = faceCascade.cascade_face_eyes(frameg)
        if eyes is None:
            #If we have no series and found no eyes, we can't proceed
            cv2.imshow(name,frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            continue
        cv2.rectangle(frame,(eyes[0][0],eyes[0][1]),(eyes[0][0]+eyes[0][2],eyes[0][1]+eyes[0][3]),
                      (0,0,255),3)
        #Centralise the eye XY coordinates
        eyes = eyes[1:,:2]+eyes[1:,2:]/2
        #Those are our tracks
        corners = eyes.astype(int)

    #Track and draw the tracking corners
    for i in range(2):
        ct,cimgs[i],e = utils.locmin(frameg,corners[i][0],corners[i][1],trackradius,cimgs[i],kernelsize)
        trackerror = utils.smooth(trackerror,np.array([e]),0.5)
        corners[i] = utils.smooth(corners[i],ct,trackingsmoothfactor)
        cv2.circle(frame,tuple(corners[i]),4,(0,0,255),5)
        #if type(cimgs[i])!=type(None):
            #cv2.imshow(str(i),cimgs[i])

    #Check if our current track is an outlier (and move our series forwards in time)
    outlier,eyeSeries,ratio = utils.point_is_outlier(trackerror,eyeSeries,eyeSeriesSensitivity,eyeSeriesLength,True)    
    #You can use this line to adjust the outlier sensitivity
    #print(ratio)
    if outlier or np.all(corners[0]==corners[1]):
        #Reset our series so we look for a new track
        eyeSeries = None

    #Calculate the 3d position
    if usematrix:
        pos_3d = utils.smooth(pos_3d,utils.real_xyz_from_screen_xy_mtx(corners,np.array(frame.shape[:2][::-1])/2,
                                        eyeseparation,iimatrix),smoothingf)
    else:
        pos_3d = utils.smooth(pos_3d,utils.real_xyz_from_screen_xy(corners,np.array(frame.shape[:2][::-1])/2,
                                        separation_at_ref,reference_distance,
                                        eyeseparation),smoothingf) 
    
    #Generate an image that's fixed at the person's position on the "other
    # side" of the camera and at an inverted z
    pic = utils.project_image_at_xyz(llyn,np.array([pos_3d[0],pos_3d[1],pos_3d[2]]),(640,480),reference_distance,2,(0,0),screen_pixel_density)
    
    #Position visualisation
    proj = np.zeros((480,640))
    cv2.circle(proj,(10,240),4,(255,255,255),3)
    cv2.putText(proj,"Cam",(10,250),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
    projscale=5
    personproj = np.array([projscale*pos_3d[2],-projscale*pos_3d[1],30-30*(pos_3d[0]+100)/200]).astype(int)
    cv2.circle(proj,(personproj[0],personproj[1]+240),personproj[2],(255,255,255),3)

    cv2.putText(frame,f"Sep.: {int(np.linalg.norm(corners[0]-corners[1]))}px",(10,10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    cv2.imshow(name,frame)
    cv2.imshow(name2,proj)
    cv2.imshow(name3,pic)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        eyeSeries=None
    elif key == ord('m'):
        usematrix = not usematrix
cv2.destroyAllWindows()
cap.release()