import os
import cv2
import time

start = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Rostros'
image_dir = os.path.join(BASE_DIR,faces_path)

current_id=0

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('mp4'):
            path = os.path.join(root, file)
            cap = cv2.VideoCapture(path)
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.endswith('mp4'):
                        count = 20
                        while(count>0):
                            ret, frame = cap.read()
                            name=root+'\\frame'+str(count)+'.png'
                            cv2.imwrite(name,frame)
                            count = count - 1

end = time.time()
elapsedTime=end-start
print('Elapsed time: ', elapsedTime)