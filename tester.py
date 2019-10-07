import cv2
import os
import numpy as np
import svmf as sv
import time

test_img=cv2.imread('C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Rostro.png') #Rostro.png, cedula_frontal.png
start = time.time()
faces_detected,gray_img = sv.faceDetection(test_img)
print('faces detected: ', faces_detected)

#for (x,y,w,h) in faces_detected:
#    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

#resized_img=cv2.resize(test_img,(498,500))
#cv2.imshow('face detection', resized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#faces,faceID=sv.labels_for_training_data('C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Rostros')
#face_recognizer=sv.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\SVM\\trainingData.yml')
name={0:'Diego',1:'Luis_Miguel',2:'Pablo_Esteban'}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray=gray_img[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(roi_gray)   #confidence: the calculated distance between histograms
    print('confidence: ',confidence)
    print('label: ',label)
    sv.draw_rect(test_img,face)
    predicted_name=name[label]
    if confidence >= 65:
        continue
    sv.put_text(test_img,predicted_name,x,y)
end = time.time()
elapsedTime=end - start
print('Elapsed time:', elapsedTime)
resized_img=cv2.resize(test_img,(498,500))
cv2.imshow('face detection', resized_img)
cv2.imwrite('face_detected.png',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
