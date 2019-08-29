import cv2

camera = cv2.VideoCapture(0)

while True:
    ret,img = camera.read()
    
    if ret==False:
        continue
        
   
    face_detector = cv2.CascadeClassifier('FaceCascade/templatedata.xml')
    faces = face_detector.detectMultiScale(img,1.3,5)
    
    for f in faces:
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

    cv2.imshow("SomeTitle",img)
    cv2.waitKey(1)
    
    