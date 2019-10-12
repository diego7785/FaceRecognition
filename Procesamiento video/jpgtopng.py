import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Rostros'
image_dir = os.path.join(BASE_DIR,faces_path)
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg')  or file.endswith('JPG') or file.endswith('jpeg'):
            path = os.path.join(root, file)
            img = cv2.imread(path)
            name = file[:-4] + '.png'
            keep=root+'\\'+name
            cv2.imwrite(keep,img)