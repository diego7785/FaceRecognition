import os
import cv2
import numpy as np
from time import time
from skimage import feature

#Tiempo de inicio.
star_time = time()

#Cargar imagen 1
imagen = cv2.imread("/home/johan/Pruebas/Caras/brayan.png")

#Gray Scale
gray = cv2.cvtColor(imagen.copy(), cv2.COLOR_BGR2GRAY)

#Lista descriptores para imagen 1
hogList = []
lbpList = []

#Modificar tamaño de imagen para acelerar el proceso
imagen = cv2.resize(imagen, (64, 128))

#HOG de skimage
descriptorHog, hogimage = feature.hog(imagen,
                                    orientations=9,
                                    pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2),
                                    transform_sqrt=True,
                                    block_norm="L1",
                                    visualize=True)

#Agregar descriptor a la lista correspondiente
hogList.append(descriptorHog)

# Descriptor LBP
lbp = feature.local_binary_pattern(gray.copy(), 24, 3, method="uniform")

#Usaré hist como descriptor
(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

#Agregando 'descriptor' a la lista
lbpList.append(hist)

#Imagen a comprobar
comprobar = cv2.imread("/home/johan/Pruebas/Caras/caraDaniel.png")

#Gray Scale
gray = cv2.cvtColor(comprobar.copy(), cv2.COLOR_BGR2GRAY)

#Lista de descriptores imagen2
hogListComprobar = []
lbpListComprobar = []

#Modificar tamaño para acelerar proceso. IMPORTANTE: debe quedar del mismo tamaño a imagen 1
comprobar = cv2.resize(comprobar, (64, 128))

#HOG imagen a comprobar
descriptorHog, hogimage = feature.hog(comprobar,
                                    orientations=9,
                                    pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2),
                                    transform_sqrt=True,
                                    block_norm="L1",
                                    visualize=True)

#Agregando descriptor a la lista
hogListComprobar.append(descriptorHog)

#LBP descriptor
lbp = feature.local_binary_pattern(gray.copy(), 24, 3, method="uniform")

#Usaré hist como descriptor
(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

#Agregando 'descriptor'
lbpListComprobar.append(hist)

#Resta entre descriptor HOG imagen1 e imagen2
diferenciaHog = []
for x in range(0, len(hogList)):
    diferenciaHog = hogList[x] - hogListComprobar[x]

#Resta entre descriptor LBP imagen1 e imagen2
diferenciaLbp = []
for x in range(0, len(lbpList)):
    diferenciaLbp = lbpList[x] - lbpListComprobar[x]

#volverlo binario
contador = 0
while(contador < len(diferenciaHog)):
    if diferenciaHog[contador] < 0.05 and diferenciaHog[contador] > -0.05:
        diferenciaHog[contador] = 1
    else:
        diferenciaHog[contador] = 0
    contador+=1

contador = 0
while(contador < len(diferenciaLbp)):
    if diferenciaLbp[contador] < 0.01 and diferenciaLbp[contador] > -0.01:
        diferenciaLbp[contador] = 1
    else:
        diferenciaLbp[contador] = 0
    contador+=1

#Contar 1's en las listas para usarlos como acertados
contadorHog = 0
for x in range(0, len(diferenciaHog)):
    contadorHog += diferenciaHog[x]

contadorLbp = 0
for x in range(0, len(diferenciaLbp)):
    contadorLbp += diferenciaLbp[x]


print("Tamaño HOG: ", len(diferenciaHog))
print(" Acertados: ", str(contadorHog))
print("Tamaño LBP: ", len(diferenciaLbp))
print(" Acertados: ", str(contadorLbp))

#Usaré 1/4 del tamaño del descriptor como humbral
fraccionHOG = contadorHog // 4

#humbral
humbralHOG = len(diferenciaHog) - fraccionHOG
humbralLBP = len(diferenciaLbp) - 2

print()

print("HumbralHOG: ", humbralHOG)
print("HumbralLBP: ", humbralLBP)

print()

if contadorHog > humbralHOG and contadorLbp >= humbralLBP:
    print("Aprobado")
else:
    print("No coincide")

elapsed_time = time() - star_time

print()

print("Tiempo: ", elapsed_time)