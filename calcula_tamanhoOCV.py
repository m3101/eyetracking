import cv2
import numpy as np
import math
#Camera matrix
with open("inverseIntrinsicMatrix.np","rb") as f:
    im = np.load(f)
    print(im)

#Tracks the local minimum
cv2.namedWindow("t",cv2.WINDOW_NORMAL)
def locmin(img,x,y,w,h):
    if x<w or img.shape[1]-x<w or y<h or img.shape[0]-y<h:
        return (x,y)
    area = img[y-h:y+h,x-w:x+w]
    m = np.unravel_index(np.argmin(area),area.shape)
    return (x+(int(m[1]-w)),y+(int(m[0]-h)))
#Corners of the tracking area
corners = [(0,0),(0,0),(0,0),(0,0)]

name = "Rastreamento"
cv2.namedWindow(name)

cI=0
def trackclick(event, x, y, flags, param):
    global corners,cI
    if event == cv2.EVENT_LBUTTONDOWN:
        #Define a posição do rastreamento
        corners[cI]=(x,y)
        print(f"Corner {cI} is now at {corners[cI]}")
        cI+=1
        cI%=2

cv2.setMouseCallback(name, trackclick)

cap = cv2.VideoCapture(0)

trackradius=1
while(True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

    #Track and draw the tracking corners
    for i in range(4):
        corners[i]=locmin(frame[:,:,2],corners[i][0],corners[i][1],10,10)
        cv2.circle(frame,corners[i],4,(0,0,255),5)
    
    #Calculate the vectors for the top-leftmost and bottom-rightmost corners
    d1 = np.dot(im,np.insert(corners[0],2,1))
    d1 = d1/np.linalg.norm(d1)
    d2 = np.dot(im,np.insert(corners[1],2,1))
    d2 = d2/np.linalg.norm(d2)
    #This assumes the points are perpendicular
    #Calculates their angle and measures the z distance in the world space
    theta = math.acos(np.dot(d1,d2))
    realh = 6
    try:
        z = realh/(2*math.tan(theta/2))
    except:
        z=0
    cv2.putText(frame,f"Theta = {round(theta,4)}",(0,100),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    cv2.putText(frame,f"Z = {round(z,4)}",(0,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    cv2.putText(frame,f"d1 = {d1}",(0,150),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    cv2.putText(frame,f"d2 = {d2}",(0,200),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)


    cv2.imshow(name,frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()