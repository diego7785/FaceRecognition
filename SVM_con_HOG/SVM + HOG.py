from skimage import feature
from time import time
from pathlib import Path
import cv2
import os
import numpy as np

start_time = time()
jeje = ""
video = cv2.VideoCapture("video5.mp4")
videoFalso = cv2.VideoCapture("videoFalso.mp4")
path = "/home/johan/Downloads/crop_part1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
negatives_dir = os.path.join(BASE_DIR,path)

#fps por segundo
#fps = int(video.get(cv2.CAP_PROP_FPS))
#fpsTotales = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#Archivo que reconoce el rostro
cascPath = "haarcascade_frontalcatface_extended.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

imagenes = []
labels = []
counter1 = 0

while(True):
    if counter1 == 3:
        counter1 = 0
        ret, frame = video.read()

        if ret == True:
            frame = cv2.resize(frame, (500, 500))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                frame = frame[y:y+h, x:x+w]

            try:
                cv2.imshow('hermoso', frame)
                cv2.waitKey(0)
                frame = cv2.resize(frame, (64, 64))

                descriptor, hogimage = feature.hog(frame,
                                                   orientations=9,
                                                   pixels_per_cell=(8, 8),
                                                   cells_per_block=(2, 2),
                                                   transform_sqrt=True,
                                                   block_norm="L1",
                                                   visualize=True)
                print("estoy en hog mano")
                imagenes.append(descriptor)
                labels.append(1)

            except Exception as e:
                print("Ocurrió un error con una de las fotos")

        else:
            break
    else:
        counter1+=1
        ret, frame = video.read()
        if ret == True:
            frame = cv2.resize(frame, (64, 64))

video.release()

counter1 = 0

while(True):
    ret, frame = videoFalso.read()

    if ret == True:
        frame = cv2.resize(frame, (500, 500))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            frame = frame[y:y+h, x:x+w]

        try:
            frame = cv2.resize(frame, (64, 64))

            descriptor, hogimage = feature.hog(frame,
                                                orientations=9,
                                                pixels_per_cell=(8, 8),
                                                cells_per_block=(2, 2),
                                                transform_sqrt=True,
                                                block_norm="L1",
                                                visualize=True)
            print("estoy en hog #2 mano")
            imagenes.append(descriptor)
            labels.append(-1)

        except Exception as e:
            print("Ocurrió un error con una de las fotos")

    else:
        break

videoFalso.release()
'''
for root, dirs, files in os.walk(negatives_dir):
    for file in files:
        if file.endswith('jpg'):
            path = os.path.join(root,file)
            image = cv2.imread(path)
            image = cv2.resize(image, (64, 64))
            descriptor, hogimage = feature.hog(image,
                                               orientations=9,
                                               pixels_per_cell=(8, 8),
                                               cells_per_block=(2, 2),
                                               transform_sqrt=True,
                                               block_norm="L1",
                                               visualize=True)
            print("estoy en hog #2 mano")

            imagenes.append(descriptor)
            labels.append(-1)
'''
descriptorHog = np.array(imagenes, np.float32)
response = np.array(labels, np.int)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(descriptorHog, cv2.ml.ROW_SAMPLE, response)
svm.save('svm_trained.dat')

#pruebas
imagen = cv2.imread("caraAndres.png")
imagen = cv2.resize(imagen, (64, 64))

descriptor, hogimage = feature.hog(
    imagen,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    transform_sqrt=True,
    block_norm="L1",
    visualize=True)

test = cv2.ml.SVM_load('svm_trained.dat')

prueba = []
prueba.append(descriptor)
prueba = np.array(prueba, np.float32)

resultado = test.predict(prueba)
print(resultado)


elapsed_time = time() - start_time

print(elapsed_time)