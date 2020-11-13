#Bismillahi Roxmaniy Roxiym

# import stuff
import cv2
import keras
import numpy as np

# restore the pre-trained model for mask prediction
#mask_model = keras.models.load_model('face_mask.h5')

import math

cap = cv2.VideoCapture(0)
frameRate = cap.get(1) #frame rate
i = 0
j = 0
while(cap.isOpened()):  
    i += 1
    frameId = cap.get(30) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (i % 30 == 0):
        j += 1
        cv2.imwrite("Face_images/Jinju/pict%i.jpg" %j, frame)
    #display the resulting frame
    cv2.imshow('Webcam Live Face Mask Detection | Jinju Seo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

테스트 하는중입니다.
