#Extraer los rostros de videos descargados orientados a depresión o ansiedad
import os
import cv2
import imutils

estado_emocional = 'Inestable' #Inestable: Depresion o ansiedad / Estable: No depresion ni ansiedad

direccion_video = r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Videos\Ansiedad\video4.mp4' #Ruta del video a extraer los fotogramas
#Acomodar la ruta del video a extraer los fotogramas
direccion_video = direccion_video.replace('\\','/') #Reemplaza las diagonales invertidas por diagonales normales

carpetaFotogramas = 'C:/Users/carolina/Documents/VS Code/Reconocimiento_emociones_modelo/Fotogramas' #Ruta donde se almacenarán los fotogramas
sucarpeta_estado_emocional_fotogramas = carpetaFotogramas + '/' + estado_emocional #Carpeta donde se almacenarán los fotogramas del estado emocional seleccionado

if not os.path.exists(sucarpeta_estado_emocional_fotogramas): #Si no existe la carpeta la crea
    print('Carpeta creada: ',sucarpeta_estado_emocional_fotogramas) #Muestra mensaje de que la carpeta fue creada
    os.makedirs(sucarpeta_estado_emocional_fotogramas) #Crea la carpeta para almacenar las emociones

n_imagenes_carpeta = len(os.listdir(sucarpeta_estado_emocional_fotogramas)) #Cuenta el número de imágenes en la carpeta
print('Número de imágenes en la carpeta: ',n_imagenes_carpeta) #Muestra el número de imágenes en la carpeta
contador_fotogramas = n_imagenes_carpeta #Contador de fotogramas

cap = cv2.VideoCapture(direccion_video) #Carga el video a analizar

detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de Haar para detección de rostros

while True:
    
        ret, frame = cap.read() #Lee el fotograma del video
        if ret == False: break #Si no hay más fotogramas termina el ciclo
        frame =  imutils.resize(frame, width=640) #Redimensiona el fotograma
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convierte el fotograma a escala de grises
        auxFrame = frame.copy() #Copia del fotograma
    
        rostros = detector_rostros.detectMultiScale(gray,1.3,5) #Detecta los rostros en el fotograma
    
        for (x,y,w,h) in rostros:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #Dibuja un rectángulo alrededor del rostro
            rostro = auxFrame[y:y+h,x:x+w] #Extrae el rostro del fotograma
            rostro = cv2.resize(rostro,(48,48),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(sucarpeta_estado_emocional_fotogramas + '/rostro_{}.jpg'.format(contador_fotogramas),rostro) #Guarda el rostro en la carpeta correspondiente
            contador_fotogramas = contador_fotogramas + 1 #Incrementa el contador de fotogramas
    
        cv2.imshow('frame',frame) #Muestra el fotograma
    
        k =  cv2.waitKey(1) #Espera una tecla para continuar
        if k == 27 or contador_fotogramas >= n_imagenes_carpeta + 300: #Si se presiona la tecla Esc termina el ciclo
            break

cap.release() #Libera el video
cv2.destroyAllWindows() #Cierra todas las ventanas
