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
# Measure your pupil separation in real units
# separation_at_ref can be obtained using the tracker
# Stay at a real reference_distance, track your eyes
# and register the pixel separation (see the top left corner of the
# tracker screen)
eyeseparation=6#cm
reference_distance=30#cm
separation_at_ref=116#pixels
#For projecting
#Measure the projecting window width
# in pixels divided by its real width
# in real units
screen_pixel_density = 640/16 ##pixels/cm

import cv2
import numpy as np
import math
import utils
import pdb

#Pupil positions (and kernels/images)
corners = np.array([[0,0],[0,0]])
cimgs = [None,None]

#Windows
name = "Tracker"
cv2.namedWindow(name,cv2.WINDOW_NORMAL)

name2 = "Projection"
cv2.namedWindow(name2,cv2.WINDOW_NORMAL)

name3 = "3d static image"
cv2.namedWindow(name3,cv2.WINDOW_AUTOSIZE)

#cI defines what corner will be updated with this click
cI=0
def trackclick(event, x, y, flags, param):
    global corners,cI
    if event == cv2.EVENT_LBUTTONDOWN:
        #Defines the current corner's position
        corners[cI]=(x,y)
        cimgs[cI]=None
        print(f"Corner {cI} is now at {corners[cI]}")
        cI+=1
        cI%=2

cv2.setMouseCallback(name, trackclick)

cap = cv2.VideoCapture(0)

#Our images to project
llyn = cv2.imread("llyn.jpg")

#Smoothing factors
smoothingf = 0.8
trackingsmoothfactor = 0.4

#Tracking radii
trackradius = 10
kernelsize = 5

#Final coordinates
pos_3d = np.zeros(3)

#Main loop
while(True):
    #Caputre and process the frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame,1)
    #Grayscale and median blur for lessening noise
    frameg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frameg = cv2.medianBlur(frameg,7)
    #Track and draw the tracking corners
    for i in range(2):
        ct,cimgs[i] = utils.locmin(frameg,corners[i][0],corners[i][1],trackradius,cimgs[i],kernelsize)
        corners[i] = utils.smooth(corners[i],ct,trackingsmoothfactor)
        cv2.circle(frame,tuple(corners[i]),4,(0,0,255),5)
        #if type(cimgs[i])!=type(None):
            #cv2.imshow(str(i),cimgs[i])
    
    #Calculate the 3d position
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
        running=False
        break
    if key == ord('r'):
        corners = np.array([(0,0),(0,0),(0,0),(0,0)])
        cI=0
cv2.destroyAllWindows()
cap.release()