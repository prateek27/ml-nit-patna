import cv2
import numpy as np


# Init Camera
camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier("FaceCascade/templatedata.xml")
dataset_path = "./pics/"

face_data = []
cnt = 0

filename = input("Enter name of person ")
while True:
    ret, img = camera.read()
    if ret==False:
        continue
    
    faces = face_detector.detectMultiScale(img,1.3,5)
    if(len(faces)==0):
        continue
    
    face=faces[0]
    x,y,w,h = face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,200,0),5)
    cropped_img = img[y:y+h,x:x+w,:]
    cropped_img = cv2.resize(cropped_img,(100,100))
    cv2.imshow("Image",img)
    cv2.imshow("Cropped Img",cropped_img)
    cnt += 1
    if cnt%10==0:
        face_data.append(cropped_img)
        print("Pics clicked ",len(face_data))
    
    cv2.waitKey(1)
    if cnt==200:
        break
        
face_data = np.asarray(face_data)
print(face_data.shape)
np.save(filename+".npy",face_data)
        
    
    
    