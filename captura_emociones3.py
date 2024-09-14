import cv2
import os
import imutils
import numpy as np

emotionName = 'Enojo'
dataPath = 'C:/Users/carolina/Documents/VS Code/Reconocimiento_emociones_modelo/Data_2'
emotionsPath = os.path.join(dataPath, emotionName)

if not os.path.exists(emotionsPath):
    print('Carpeta creada: ', emotionsPath)
    os.makedirs(emotionsPath)

initial_count = len(os.listdir(emotionsPath))
count = initial_count

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar los clasificadores de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def check_image(image, name):
    print(f"Comprobando {name}:")
    print(f"  Forma: {image.shape}")
    print(f"  Tipo de datos: {image.dtype}")
    print(f"  Valores mínimo y máximo: {np.min(image)}, {np.max(image)}")
    if image.ndim == 3:
        print(f"  Canales: {image.shape[2]}")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break
    
    frame = imutils.resize(frame, width=640)
    frame_count += 1
    if frame_count % 30 == 0:
        check_image(frame, "Frame original")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_count % 30 == 0:
        check_image(gray, "Frame en escala de grises")
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if frame_count % 30 == 0:
        print(f"Caras detectadas: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        rostro = frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(emotionsPath, f'rostro_{count}.jpg'), rostro)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= initial_count + 200:
        break

cap.release()
cv2.destroyAllWindows()