import os
import cv2
import time
import subprocess
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import threading

start = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\VideoBen'
image_dir = os.path.join(BASE_DIR,faces_path)
'''
#PERFORMS CONVERTION
current_id=0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('MOV'):
            path = os.path.join(root, file)
            name=file[:-4]+'.mp4'
            #print(name.replace(" ", "-"))
            path_save= os.path.join(root,name)
            print(path)
            print(path_save)
            converter = 'ffmpeg -i '
            converter= converter + path.replace('\\','\\\\')
            converter=converter+' '
            converter=converter+ path_save.replace('\\', '\\\\')
            #os.system(converter)
            cmds = ['ffmpeg', '-i', path, path_save]
            #subprocess.Popen(cmds)
            #print(converter)
            subprocess.run(cmds, shell=True)
'''
'''
video_path='C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\VideoAndresH\\video.mp4'

#THREADS FRAME EXTRACTION
def worker(start, end,cap):
    count = start
    while (count < end):
        cap.set(start, end)
        ret, frame = cap.read()
        name = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\AndresHRostro' + '\\frame' + str(count) + '.png'
        cv2.imwrite(name, frame)
        count+=1
    cap.release()
    #SEPARATE-----------------------------------------------
    prev_time = -1
    while True:
        grabbed = cap.grab()
        if grabbed:
            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if int(time_s) > int(prev_time):
                # Only retrieve and save the first frame in each new second

                #name = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\AndresHRostro' + '\\frame' + str(prev_time) + '.png'
                #cv2.imwrite(name,cap.retrieve())
                prev_time = time_s
        else:
            break


cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fpsTotales = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tiempo = fpsTotales / fps
end = int(fpsTotales/3)
t = threading.Thread(target=worker,args=(1,end,cap))
t1 = threading.Thread(target=worker,args=(end+1,end*2,cap))
t3 = threading.Thread(target=worker, args=((end*2)+1,(end*3),cap))
t.start()
t1.start()
t3.start()
'''
#FRAME EXTRACTION WITH QUEUES (faster)
video_path='C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\VideoBen\\video.MOV'
print("[INFO] starting video file thread...")
fvs = FileVideoStream(video_path).start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()
count=0
# loop over frames from the video file stream
while fvs.more():
    if(count>80):
        break
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    # show the frame and update the FPS counter
    name = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\BenRostro' + '\\frame' + str(count) + '.png'
    cv2.imwrite(name, frame)
    fps.update()
    count += 1

    # stop the timer and display FPS information
    fps.stop()

    # do a bit of cleanup
    fvs.stop()

'''
#NORMAL FRAME EXTRACTION (SLOW)
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('mp4'):
            path = os.path.join(root, file)
            cap = cv2.VideoCapture(path)
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.endswith('mp4'):
                        # fps por segundo
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        fpsTotales = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        tiempo = fpsTotales / fps
                        count = 1
                        while(count<fpsTotales*7):
                            cap.set(count,fps*3)
                            ret, frame = cap.read()
                            name='C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\BenRostro'+'\\frame'+str(count)+'.png'
                            cv2.imwrite(name,frame)
                            count = count + fps*2

            cap.release()
'''
end = time.time()
elapsedTime=end-start
print('Elapsed time: ', elapsedTime)
