import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from time import sleep

from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import model_from_json

import tensorflow as tf

json_file = open(r'E:\ML Project\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from tensorflow.keras.models import model_from_json
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(r'E:\ML Project\model.h5')
print("\n \n")
print("*********************** Model is loaded from disk ***********************")

    

#face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



class_labels = ['with mask','without mask']

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(200,200),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = loaded_model.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Face Mask Detector',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()