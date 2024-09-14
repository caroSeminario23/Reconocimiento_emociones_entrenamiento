import cv2
import os
import imutils
import dlib
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

predictor_path = "C:/Users/carolina/Documents/VS Code/Reconocimiento_emociones_modelo/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

def check_image(image, name):
    print(f"Comprobando {name}:")
    print(f"  Forma: {image.shape}")
    print(f"  Tipo de datos: {image.dtype}")
    print(f"  Valores mínimo y máximo: {np.min(image)}, {np.max(image)}")
    if image.ndim == 3:
        print(f"  Canales: {image.shape[2]}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break
    
    frame = imutils.resize(frame, width=640)
    check_image(frame, "Frame original")
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    check_image(rgb_frame, "Frame en RGB")
    
    try:
        faces = face_detector(rgb_frame)
        print(f"Caras detectadas: {len(faces)}")
    except Exception as e:
        print(f"Error al detectar caras: {e}")
        continue

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        rostro = frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(emotionsPath, f'rostro_{count}.jpg'), rostro)
        count += 1

        try:
            landmarks = landmark_predictor(rgb_frame, face)
            print(f"Landmarks detectados para la cara en ({x}, {y}, {w}, {h})")
            for n in range(68):
                x_point = landmarks.part(n).x
                y_point = landmarks.part(n).y
                cv2.circle(frame, (x_point, y_point), 1, (0, 0, 255), -1)
        except RuntimeError as e:
            print(f"Error al detectar landmarks: {e}")
            continue

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= initial_count + 200:
        break

cap.release()
cv2.destroyAllWindows()