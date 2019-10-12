#This one works
#IT'S MANDATORY TO RESIZE THE IMAGE BEFORE HOG
import cv2
import numpy as np
import os
from skimage import feature
#LBP
from sklearn.svm import LinearSVC
import time

start = time.time()
'''
#FRAMES EXTRACTION FROM VIDEO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Videos'
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
                            name='C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\DiegoRostro'+'\\frame'+str(count)+'.png'
                            cv2.imwrite(name,frame)
                            count = count - 1
'''
#Path to the extracted frames directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#faces_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\PabloEstebanRostro'  #CON EL FUNCIONA
faces_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\FotosPabloEsteban'
negatives_path = 'C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\No'
image_dir = os.path.join(BASE_DIR,faces_path)
negatives_dir = os.path.join(BASE_DIR,negatives_path)

#Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#HOG descriptor
hog=cv2.HOGDescriptor((64,128),(16,16),(8,8),(8,8),9)

training =[]
labels=[]
#Positive files search
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png'):
            path = os.path.join(root, file)
            image = cv2.imread(path,0)
            h,w=image.shape
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=9, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x,y,w,h) in faces:
                image = image[y:y+h,x:x+w]
            #cv2.imshow('imagen',image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            image=cv2.resize(image,(64,128),interpolation=cv2.INTER_CUBIC)
            #HOG
            #hist=hog.compute(image)
            #training.append(hist)
            #labels.append(1)
            #LBP
            lbp = feature.local_binary_pattern(image, 24, 3, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 24 + 3),range=(0, 24 + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            training.append(hist)
            labels.append(1)

#Negative files search
for root, dirs, files in os.walk(negatives_dir):
    for file in files:
        if file.endswith('png'):
            path = os.path.join(root,file)
            image = cv2.imread(path, 0)
            resized_image = cv2.resize(image, (64,128))
            # HOG
            #hist=hog.compute(resized_image)
            #training.append(hist)
            #labels.append(-1)
            # LBP
            lbp = feature.local_binary_pattern(resized_image, 24, 3, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            training.append(hist)
            labels.append(-1)

#HOG
#featuresList=np.array(training,np.float32)  #Training set
#labelsList=np.array(labels,np.int)   #Labels set
#svm = cv2.ml.SVM_create()
#svm.setType(cv2.ml.SVM_C_SVC)
#svm.setKernel(cv2.ml.SVM_LINEAR)
#a = np.asarray(featuresList)
#b = np.array(labelsList)
#svm.train(a, cv2.ml.ROW_SAMPLE, b)
#svm.save('hog_svm.dat')
#Testing
#test = cv2.ml.SVM_load("hog_svm.dat")
#image_test = cv2.imread('C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\pablo_esteban1.png',0)  #FUNCIONA CON SUS SELFIES Y SU DOCUMENTO
#image_test = cv2.imread('C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Cedulas\\cedula_Diego1.png',0)
#image_test = cv2.imread('C:\\Users\\Usuario\\Downloads\\diego.jpg',0)
#image_test=cv2.resize(image_test, (698, 452))
#faces = face_cascade.detectMultiScale(image_test, scaleFactor=1.14, minNeighbors=9, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#for (x, y, w, h) in faces:
#    image_test = image_test[y:y + h, x:x + w]
#cv2.imshow('face',image_test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#image_test=cv2.resize(image_test,(64,128),interpolation=cv2.INTER_CUBIC)
#HOG
#h_test=hog.compute(image_test)
#test_=list()
#test_.append(h_test)
#test_=np.array(test_)
#result=test.predict(test_)
#print(result[1])
#end = time.time()
#elapsedTime = end-start
#print('elapsed time:',elapsedTime)

#LBP
testing=[]
model = LinearSVC(C=100.0, random_state=42)
model.fit(training, labels)
image_test = cv2.imread('C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Rostros\\PabloEsteban.png',0)
#image_test=cv2.resize(image_test, (698, 582))
#cv2.imshow('cedula',image_test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
faces = face_cascade.detectMultiScale(image_test, scaleFactor=1.14, minNeighbors=9, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
iterator=0
for (x, y, w, h) in faces:
    image_test = image_test[y:y + h, x:x + w]
    iterator+=1

#cv2.imshow('face',image_test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
lbp = feature.local_binary_pattern(image_test, 24, 3, method="uniform")
(h_test, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 24 + 3),range=(0, 24 + 2))
h_test = h_test.astype("float")
h_test /= (h_test.sum() + 1e-7)
testing.append(h_test)
prediction = model.predict(testing)
print(prediction)
end = time.time()
elapsedTime = end-start
print('elapsed time:',elapsedTime)








'''
#FUNCIONA PERO TODAVÍA NO SÉ CÓMO
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize

def load_image_files(container_path, dimension=(64, 64)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
           # if file.endswith('png')    #VERIFICATION NEEDED
                img = skimage.io.imread(file)
                img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
                flat_data.append(img_resized.flatten())  #.flatten() NORMALIZATION?
                images.append(img_resized)
                target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

image_dataset = load_image_files("./Rostros")

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid, cv=3, iid=True)
clf.fit(X_train, y_train)

#test_list=[]
#img_test = skimage.io.imread('./cedula_frontal.png')
#img_test_resized = resize(img_test, (64,64), anti_aliasing=True, mode='reflect')
#i=len(y_test)
#while(i>0):
# test_list.append(img_test_resized.flatten())
# i=i-1
#X_test = np.array(test_list)

y_pred = clf.predict(X_test)

print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))

'''
'''
import cv2
import numpy as np
import glob
import os
from skimage.feature import hog

positive_images_path='C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Rostros'
negative_images_path='C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\No'

samples = []
labels = []

# Get positive samples
for filename in glob.glob(os.path.join(positive_images_path, '*.png')):
    img = cv2.imread(filename, 0)
    hist = hog(img, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(2, 2),block_norm='L1', visualize=False,transform_sqrt=False,feature_vector=True)
    samples.append(hist)
    labels.append(1)

# Get negative samples
for filename in glob.glob(os.path.join(negative_images_path, '*.png')):
    img = cv2.imread(filename, 0)
    hist = hog(img, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(2, 2),block_norm='L1', visualize=False,transform_sqrt=False,feature_vector=True)
    samples.append(hist)
    labels.append(0)

# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF) # cv2.ml.SVM_LINEAR
svm.setGamma(5.383)
svm.setC(2.67)
print(samples)
svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
svm.save('C:\\Users\\Usuario\\Google Drive\\Truora\\FaceRecognition\\Rostros\\svm_data.dat')
'''

'''
# Ejemplo de deteccion facial con OpenCV y Python
# Por Glare
# www.robologs.net

import numpy as np
import cv2

# Cargamos la plantilla e inicializamos la webcam:
# !!! RECUERDA CAMBIAR EL PATH DEL ARCHIVO .xml POR EL TUYO!!!
face_cascade = cv2.CascadeClassifier('/home/glare/rostrosCV/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)

while (True):
    # Leemos un frame y lo guardamos
    valido, img = cap.read()

    # Si el frame se ha capturado correctamente, continuamos
    if valido:

        # Convertimos la imagen a blanco y negro
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Buscamos las coordenadas de los rostros (si los hay) y
        # guardamos su posicion
        array_rostros = face_cascade.detectMultiScale(img_gris, 1.3, 5)

        # Iteramos el array de rostros y pintamos un recuadro alrededor de
        # cada uno
        for (x, y, w, h) in array_rostros:
            cv2.rectangle(img, (x, y), (x + w, y + h), (125, 255, 0), 2)

        # Mostramos la imagen
        cv2.imshow('img', img)

        # Con la tecla 'q' salimos del programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2 as cv
import numpy as np
# Set up training data
labels = np.array([1, -1, -1, -1])
trainingData = np.matrix([[501, 10], [255, 10], [501, 255], [10, 501]], dtype=np.float32)
# Train the SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)
# Data for visual representation
width = 512
height = 512
image = np.zeros((height, width, 3), dtype=np.uint8)
# Show the decision regions given by the SVM
green = (0,255,0)
blue = (255,0,0)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        sampleMat = np.matrix([[j,i]], dtype=np.float32)
        response = svm.predict(sampleMat)[1]
        if response == 1:
            image[i,j] = green
        elif response == -1:
            image[i,j] = blue
# Show the training data
thickness = -1
cv.circle(image, (501,  10), 5, (  0,   0,   0), thickness)
cv.circle(image, (255,  10), 5, (255, 255, 255), thickness)
cv.circle(image, (501, 255), 5, (255, 255, 255), thickness)
cv.circle(image, ( 10, 501), 5, (255, 255, 255), thickness)
# Show support vectors
thickness = 2
sv = svm.getUncompressedSupportVectors()
for i in range(sv.shape[0]):
    cv.circle(image, (sv[i,0], sv[i,1]), 6, (128, 128, 128), thickness)
cv.imwrite('result.png', image) # save the image
cv.imshow('SVM Simple Example', image) # show it to the user
cv.waitKey()
'''
'''
# Se cargan módulos
import cv2
import argparse
import os.path
import numpy as np
import random
import Image

#img=cv2.imread('./Rostros/cedula_frontal.png')

# Se configura los argumentos de la línea de comandos
p = argparse.ArgumentParser("mostrandoimagen.py")
p.add_argument("positivos",default=None,
            action="store", help="directorio de archivo a procesar")
p.add_argument("negativos",default=None,
            action="store", help="directorio de archivo a procesar")
p.add_argument("-m","--model",default="model.svm",
            action="store", help="Modelo de SVM")
opts = p.parse_args()

listing=os.listdir(opts.positivos)
listing=["{0}/{1}".format(opts.positivos,namefile)
                        for namefile in listing if namefile.endswith('jpg')
                                                or namefile.endswith('png')]

hog = cv2.HOGDescriptor((48,48),(16,16),(8,8),(8,8),9)


trainData=[]
print("Generating samples")
for filename in listing:
    # Se abre la imagen
    img = Image.open(filename)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    des=hog.compute(img)
    trainData.append((des,1))


listing=os.listdir(opts.negativos)
listing=["{0}/{1}".format(opts.negativos,namefile)
                        for namefile in listing if namefile.endswith('jpg')
                                                or namefile.endswith('png')]

print(len(trainData))
print("Generating samples")
for filename in listing:
    # Se abre la imagen
    img = Image.open(filename)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    height, width, depth = img.shape

    for i in range(10):
        h=random.randint(0,height-48)
        w=random.randint(0,width-48)

        sample_img=img[h:h+48,w:w+48]

        des=hog.compute(sample_img)
        trainData.append((des,0))

random.shuffle(trainData)
trainData,responses=zip(*trainData)


print(len(responses))
print("Training")

trainingDataMat = np.float32(trainData) #.resize(-1,900)
labelsMat = np.float32(responses)
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labelsMat)


#svm.save(opts.model)

'''
'''
#CREATE ARBITRARY DATASETS
# importing scikit learn with make_blobs
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

# creating datasets X containing n_samples
# Y containing two classes
X, Y = make_blobs(n_samples=500, centers=2,
                  random_state=0, cluster_std=0.40)

# plotting scatters
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show()
'''


'''
#POINTS SEPARATION AND SHOW
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# matplotlib for plotting & numpy for handling arrays as we have a lot of that.

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # trains our SVM.  Optimization step.
    def fit(self, data):
        self.data = data
        # we'll create a dictionary with key as magnitude of vector w &
        # respective elements as vector w & b
        # Remember here we are looking for w with minimum magnitude & b with largest magnitude that
        # satisfy this eqn. yi(xi.w + b ) = 1
        # { ||w||: [w,b] }
        opt_dict = {}

        # we make transforms to assure that all versions of the vector are checked

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi+w.b) = 1

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,]

        # extremely expensive
        b_range_multiple = 5
        # we don't need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because of convex nature
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t= w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        #
                        # add a break later
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi) + b) >=1:
                                    found_option = False
                                    #print(xi, ':',yi*(np.dot(w_t,xi)+b)

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optmized a step.')
                else:
                    w = w- step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))

    # Predicts the value of a new featureset, after training the classifier
    # sign(x.w + b) is the result.
    def predict(self, features):
        # classification is just:
        # sign( x.w + b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        # if the classification isn't zero, and we have visualization on, we graph it
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200,marker='*', c=self.colors[classification])
        # else:
        #     print('featureset',features,'is on the decision boundary')
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w + b
        # v = x.w + b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # decision boundary hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
'''