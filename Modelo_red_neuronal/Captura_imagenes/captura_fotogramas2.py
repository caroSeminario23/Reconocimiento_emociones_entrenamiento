#Extraer los rostros de videos descargados orientados a depresión o ansiedad
import os
import cv2
import imutils

estado_emocional = 'Inestable' #Inestable: Depresion o ansiedad / Estable: No depresion ni ansiedad

direccion_video = r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Videos\Depresion\video9.mp4' #Ruta del video a extraer los fotogramas
carpetaFotogramas = 'C:/Users/carolina/Documents/VS Code/Reconocimiento_emociones_modelo/Fotogramas' #Ruta donde se almacenarán los fotogramas
sucarpeta_estado_emocional_fotogramas = carpetaFotogramas + '/' + estado_emocional #Carpeta donde se almacenarán los fotogramas del estado emocional seleccionado

if not os.path.exists(sucarpeta_estado_emocional_fotogramas): #Si no existe la carpeta la crea
    print('Carpeta creada: ',sucarpeta_estado_emocional_fotogramas) #Muestra mensaje de que la carpeta fue creada
    os.makedirs(sucarpeta_estado_emocional_fotogramas) #Crea la carpeta para almacenar las emociones

# Obtener el número más alto de archivo ya existente
imagenes_guardadas = os.listdir(sucarpeta_estado_emocional_fotogramas)
if imagenes_guardadas:
    # Buscar el número más alto en los nombres de archivo existentes (considerando nombres en formato 'rostro_x.jpg')
    numeros_guardados = [int(img.split('_')[1].split('.')[0]) for img in imagenes_guardadas if img.startswith('rostro')]
    contador_fotogramas = max(numeros_guardados) + 1  # Empieza con el siguiente número disponible
else:
    contador_fotogramas = 0  # Si no hay imágenes en la carpeta, empieza desde 0

cap = cv2.VideoCapture(direccion_video) #Carga el video a analizar

detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de Haar para detección de rostros

while True:
    ret, frame = cap.read() #Lee un fotograma del video
    if not ret:
        break #Si no hay más fotogramas, sale del bucle

    frame = imutils.resize(frame, width=640) #Redimensiona el fotograma
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convierte el fotograma a escala de grises
    rostros = detector_rostros.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)) #Detecta rostros en el fotograma

    for (x, y, w, h) in rostros: #Recorre los rostros detectados
        rostro = frame[y:y+h, x:x+w] #Extrae el rostro del fotograma
        rostro = cv2.resize(rostro, (48, 48), interpolation=cv2.INTER_CUBIC)
        nombre_fotograma = f"{sucarpeta_estado_emocional_fotogramas}/rostro_{contador_fotogramas}.jpg" #Nombre del archivo de la imagen
        cv2.imwrite(nombre_fotograma, rostro) #Guarda el rostro como una imagen
        contador_fotogramas += 1 #Incrementa el contador de fotogramas

cap.release() #Libera el video
cv2.destroyAllWindows() #Cierra todas las ventanas de OpenCV