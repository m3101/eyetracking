"""
    Position estimation socket server for UNIX IPC
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

#Whether the estimation should be made via camera matrices or
#the default proportion method
usematrix = True

import cv2
import numpy as np
import math
import utils
import pdb
import io
import _thread
import socket
import os

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

#Windows
name = "Tracker"
cv2.namedWindow(name,cv2.WINDOW_NORMAL)

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

#Smoothing factors
smoothingf = 0.8
trackingsmoothfactor = 0.4

#Tracking radii
trackradius = 10
kernelsize = 5

#Final coordinates
pos_3d = np.zeros(3)

#A simple UNIX socket so we can read this 3d
# posistion from other processes
running=True
def serverthread():
    global pos_3d,running
    #Clean up previous socketfiles
    try:
        os.remove("./socketfile")
    except:
        pass
    #Set up a buffer for serialisation
    buffer = io.BytesIO()

    #Bind a listener to the socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind("./socketfile")
    sock.listen(1)
    print("UNIX IPC socket open. Waiting for position requests.")
    while running:
        #Accept a connection
        connection, _ = sock.accept()
        print("Connection established.")
        #While we're still receiving position requests
        while connection.recv(1) and running:
            #Serialise the vector
            buffer.seek(0)
            buffer.flush()
            np.save(buffer,pos_3d)
            #Send its length
            connection.sendall(bytes([buffer.tell()]))
            buffer.seek(0)
            #Send the vector
            connection.sendall(buffer.read())
        connection.close()
        print("Connection closed.")
    print("Closing socket")
    sock.close()

#Start the UNIX socket server
_thread.start_new_thread(serverthread,())

#Main loop
while(True):
    #Caputre and process the frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    #frame = cv2.flip(frame,1)
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
    if usematrix:
        pos_3d = utils.smooth(pos_3d,utils.real_xyz_from_screen_xy_mtx(corners,np.array(frame.shape[:2][::-1])/2,
                                        eyeseparation,iimatrix),smoothingf)
    else:
        pos_3d = utils.smooth(pos_3d,utils.real_xyz_from_screen_xy(corners,np.array(frame.shape[:2][::-1])/2,
                                        separation_at_ref,reference_distance,
                                        eyeseparation),smoothingf)
    
    cv2.imshow(name,frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running=False
        break
    elif key == ord('r'):
        corners = np.array([(0,0),(0,0),(0,0),(0,0)])
        cI=0
    elif key == ord('m'):
        usematrix = not usematrix
cv2.destroyAllWindows()
cap.release()