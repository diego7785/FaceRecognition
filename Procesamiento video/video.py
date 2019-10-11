import os
import cv2
import time
import subprocess

start = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\VideoBen'
image_dir = os.path.join(BASE_DIR,faces_path)

#Performs convertion 
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
            subprocess.Popen(cmds)
            #print(converter)
            #subprocess.run(converter, shell=True)
'''
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
                            name='C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\GeorgeNarvaezRostro'+'\\frame'+str(count)+'.png'
                            cv2.imwrite(name,frame)
                            count = count - 1
'''
end = time.time()
elapsedTime=end-start
print('Elapsed time: ', elapsedTime)
