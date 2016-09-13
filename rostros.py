
# coding: utf-8

# In[4]:

import numpy as np
import cv2
#img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)

rostroCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
imagen = cv2.imread('funcionarios.png') 
filtro = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) 

rostros = rostroCascade.detectMultiScale(filtro,scaleFactor = 1.2,minNeighbors = 5,minSize= (30,30),flags = cv2.CASCADE_SCALE_IMAGE)


for (x, y, w, h) in rostros:
    cv2.rectangle(imagen, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = filtro[y:y+h, x:x+w]
    roi_color = imagen[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow("Rostros encontrados", imagen) 
cv2.waitKey(0)


# In[ ]:



