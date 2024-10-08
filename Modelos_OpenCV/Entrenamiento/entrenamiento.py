import cv2
import os
import numpy as np
import time

def obtenerModelo(method,facesData,labels):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo3"+method+".xml") #Almacena el modelo obtenido en un archivo XML en la carpeta del proyecto

dataPath = 'C:/Users/carolina/Documents/VS Code/Reconocimiento_emociones_modelo/Data' #Cambia a la ruta donde hayas almacenado Data
emotionsList = os.listdir(dataPath) #Cuenta la cantidad de emociones en la carpeta Data
print('Lista de emociones: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
	emotionsPath = dataPath + '/' + nameDir

	for fileName in os.listdir(emotionsPath): 
		#print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label) #Asigna un número entero a cada emoción
		facesData.append(cv2.imread(emotionsPath+'/'+fileName,0)) #Agrega la imagen a la lista facesData
		#image = cv2.imread(emotionsPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1

#obtenerModelo('EigenFaces',facesData,labels)
#obtenerModelo('FisherFaces',facesData,labels)
obtenerModelo('LBPH',facesData,labels)