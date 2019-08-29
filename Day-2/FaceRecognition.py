#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os 
import numpy as np


# In[30]:


files = os.listdir()
d = {}
cnt = 0

X = None
Y = []

for f in files:
    if f.endswith(".npy"):
        data = np.load(f)
        d[cnt] = f[:-4]
        
        if X is None:
            X = data
            target = cnt*np.ones((data.shape[0]))
            Y.append(target)
        else:
            X = np.vstack((X,data))
            target = cnt*np.ones((data.shape[0]))
            Y.append(target)
            
        cnt += 1


# In[31]:


print(d)


# In[32]:


print(X.shape)


# In[40]:


Y = np.array(Y)
Y = Y.reshape((60,))


# In[41]:


print(Y)


# In[43]:


def dist(a1,a2):
    return np.sum((a1-a2)**2)**.5

def knn(X,Y,q,k=5):
    
    m = X.shape[0]
    l = []
    
    for i in range(m):
        d = dist(q,X[i])
        l.append((d,Y[i]))
    
    l.sort()
    l = np.array(l[:k])
    l = l[:,1]
    uniq,freq = np.unique(l,return_counts=True)
    p = np.argmax(freq)
    pred = uniq[p]
    return int(pred)


# In[ ]:


import cv2
import numpy as np


# Init Camera
camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier("FaceCascade/templatedata.xml")


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
    
    #cv2.imshow("Cropped Img",cropped_img)
    label = knn(X,Y,cropped_img,7)
    name = d[label]
    #print(name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,name,(x,y-10), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("Image",img)
    
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()      
    
    
    

