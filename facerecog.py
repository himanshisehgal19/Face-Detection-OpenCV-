# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:43:20 2020

@author: user
"""


import cv2
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yellow=cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('video capture',frame)
   
    #now wait for user to input - q and then the loop will stop
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
