"""
    Gaze direction estimation through point tracking
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
import cv2
import numpy as np
import math
import utils
import pdb

#Corners of the tracking area (and tracking images)
corners = np.array([[0,0],[0,0],[0,0]])
cimgs = [None,None,None]

#Corners for projection
p_corners = np.array([[0,0],[0,0],[0,0]])

#Windows
name = "Tracker"
cv2.namedWindow(name,cv2.WINDOW_NORMAL)

name3 = "LookingAt"
cv2.namedWindow(name3,cv2.WINDOW_NORMAL)

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
        cI%=3

cv2.setMouseCallback(name, trackclick)

cap = cv2.VideoCapture(0)

#Smoothing factors
trackingsmoothfactor = 0.4
blinkingfactor = 0.5

#Tracking radii
trackradius = 15
kernelsize = 7

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
    for i in range(3):
        ct,cimgs[i] = utils.locmin(frameg,corners[i][0],corners[i][1],trackradius,cimgs[i],kernelsize)
        corners[i] = utils.smooth(corners[i],ct,trackingsmoothfactor)
        cv2.circle(frame,tuple(corners[i]),4,(255*(i>=2),0,255*(i<2)),5)
        #if type(cimgs[i])!=type(None):
            #cv2.imshow(str(i),cimgs[i])

    cv2.imshow(name,frame)

    pic = np.zeros((480,640))
    centre = ((corners[0]+corners[1])/2)-corners[2]
    try:
        cproj = centre
        d = (cproj-p_corners[1])/(p_corners[2]-p_corners[1])
        cv2.putText(pic,f"{d}",(320,240),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        cv2.putText(pic,f"{cproj-p_corners[1]}{p_corners[1:]}",(0,480),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        cv2.circle(pic,(int(d[0]*640),int(d[1]*480)),5,(255,255,255),3)
    except Exception as e:
        pass

    cv2.imshow(name3,pic)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        corners = np.array([(0,0),(0,0),(0,0),(0,0)])
        cI=0
    elif key == ord('c'):
        p_corners[0]=centre
    elif key == ord('l'):
        p_corners[1]=centre
    elif key == ord('r'):
        p_corners[2]=centre
cv2.destroyAllWindows()
cap.release()
