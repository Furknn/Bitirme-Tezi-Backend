import time
import cv2
import tensorflow as tf
import keras
import numpy as np
IMG_SIZE=128
def detectFace(directory_name):
    print("--------------------"+directory_name+"----------------")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img_array = cv2.imread(directory_name, cv2.IMREAD_GRAYSCALE)
    new_array=img_array
    img=cv2.imread(directory_name)
    faces = face_cascade.detectMultiScale(new_array, 1.08, 4)
    model=tf.keras.models.load_model('mymodel.h5')
    imJsn={}
    facesJsn={}
    
    i=0
    
    for (x, y, w, h) in faces:
        faceJsn={}
        faceimg = img[y: y + h, x: x + w]#[ny:ny + nr, nx:nx + nr]
        lastimg = cv2.resize(faceimg, (128,128))
        detectimg = new_array[y: y + h, x: x + w]
        detectimg = cv2.resize(detectimg, (128,128))
        detectimg=np.array(detectimg).reshape (-1, IMG_SIZE, IMG_SIZE, 1)
        faceName=str(int(time.time())+i)+".png"
        path2send="faces/face-"+faceName
        path="data/detectFace/"+path2send
        pred=model.predict(detectimg)
        cv2.imwrite(path,lastimg)
        faceJsn["_id"]=(i+1)
        faceJsn["X"]=x
        faceJsn["Y"]=y
        faceJsn["SizeX"]=w
        faceJsn["SizeY"]=h
        faceJsn["dir"]=path
        idx=int(np.argmax(pred))
        faceJsn["classId"]=idx
        faceJsn["csIdPer"]=np.amax(pred)
        faceJsn["allStr"]=str(pred[:])
        facesJsn[str(i)]=faceJsn
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        i=i+1
    #TODO:upload to database
    path2send = "img-" + str (int (time.time ())) + ".png"
    path="data/detectFace/"+path2send
    cv2.imwrite(path, img)
    imJsn["orgImg"]=directory_name
    imJsn["processedImg"]=path2send
    imJsn["faces"]=facesJsn
    return str(imJsn)
