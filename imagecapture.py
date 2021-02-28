# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 09:02:24 2020

@author: user
"""


import cv2
import numpy
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
facedata=[]
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    cv2.imshow("frame",frame)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    #pick the largest face(i.e last face)
    facesec=0
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,255),2)
        #extract (crop out required face):region of intrest
        offset=10
        facesec=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        facesec=cv2.resize(facesec,(100,100))
        skip=1
        if skip%10==0:
            facedata.append(facesec)
            print(len(facedata))
    cv2.imshow('frame',frame)
    cv2.imshow('face section',facesec)
    keypressed=cv2.waitKey(1)&0xFF
    if keypressed == ord('q'):
        break
#convert our face list array into numpy array
facedata=numpy.asarray(facedata)
facedata=facedata.reshape((facedata.shape[0],-1))
print(facedata.shape)
cap.release()
cv2.destroyAllWindows()