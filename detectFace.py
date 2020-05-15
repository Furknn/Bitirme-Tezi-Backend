import numpy as np
import cv2 as cv
import os
import time
import sys
import json

def detectFace(directory_name):
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    img = cv.imread(directory_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.08, 4)
    imJsn={}
    facesJsn={}
    size=64
    i=0
    for (x, y, w, h) in faces:
            faceJsn={}
            #r = max (w, h) / 2
            #centerx = x + w / 2
            #centery = y + h / 2
            #nx = int(centerx - r)
            #ny = int(centery - r)
            #nr = int(r * 2)
            faceimg = img[y: y + h, x: x + w]#[ny:ny + nr, nx:nx + nr]
            lastimg = cv.resize(faceimg, (size, size))
            faceName=str(int(time.time())+i)+".png"
            path2send="faces/face-"+faceName
            path="data/detectFace/"+path2send
            cv.imwrite(path,lastimg)
            cv.waitKey(1)
            faceJsn["_id"]=(i+1)+10000
            faceJsn["X"]=x
            faceJsn["Y"]=y
            faceJsn["Size"]=64
            faceJsn["dir"]=path
            facesJsn[str(i)]=faceJsn
            i=i+1
    for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    path2send = "img-" + str (int (time.time ())) + ".png"
    path="data/detectFace/"+path2send
    cv.imwrite(path, img)
    imJsn["orgImg"]=directory_name
    imJsn["processedImg"]=path2send
    imJsn["faces"]=facesJsn
    return str(imJsn)
