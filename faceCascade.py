"""
Default openCV2 cascade classifier functions
Adapted from https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
"""
import numpy as np
import cv2

#Set the OPENCV_SAMPLES_DATA_PATH to the path to openCV2/samples
#Before using this script

#You can also download the cascades:
#* https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_alt.xml
#* https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml
#and put them at a local folder data/haarcascades

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
try:
    face_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml'))
    eyes_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'))
except:
    print("""
    This script needs access to the pre-trained Haar cascade lists that
    come with a full OpenCV installation.
    Please either set up the OPENCV_SAMPLES_DATA_PATH environment variable
    To point to your opencv/samples folder OR
    Download the following cascades:
    * https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_alt.xml
    * https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml
    And place them at a local ./data/haarcascades directory.
    \n""")
    exit(1)

def cascade_face_eyes(img:np.ndarray)->np.ndarray:
    """
    Finds the biggest face on the screen and its two first eyes
    Returns 
            [[facex,facey,facew,faceh],
             [eye1x,eye1y,eye1w,eye1h],
             [eye2x,eye2y,eye2w,eye2h]]
    Or None when no face or not enough eyes were detected
            img should be a greyscale image
    """
    faces = np.array(face_cascade.detectMultiScale(img))
    if len(faces) == 0:
        return None
    #Calculate the sizes**2 of the faces deteceted
    #Each face is [x,y,w,h], so faces[:,2:] is [w,h] for each face
    fsizes = (faces[:,2:]**2).sum(1)
    #Our reference face will be the biggest
    face = faces[np.argmax(fsizes)]
    #Detect eyes on the area limited by the face
    eyes = np.array(eyes_cascade.detectMultiScale(img[face[1]:face[1]+face[3],
                                                      face[0]:face[0]+face[2]]))[:2]
    if len(eyes)<2 or np.all(eyes[0]==eyes[1]):
        return None
    #Sum face[x,y] for each eye's [x,y]
    eyes = eyes+np.c_[np.repeat([face[:2]],2,0),[[0,0],[0,0]]]
    return np.array([face,eyes[0],eyes[1]])