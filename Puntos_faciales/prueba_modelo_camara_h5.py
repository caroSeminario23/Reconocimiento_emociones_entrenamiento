import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Ruta del modelo
model_path = 'Puntos_faciales/Modelos_pf/model_estable_inestable.h5'

# Imprimir la ruta absoluta del modelo
print("Ruta absoluta del modelo:", os.path.abspath(model_path))

# Cargar el modelo
try:
    model = load_model(model_path)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles  # Asegurarse de importar mp_drawing_styles

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

print("Cámara iniciada correctamente.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el cuadro de la cámara.")
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar la malla facial en el rostro
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Extraer los puntos faciales
            points = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                z = landmark.z * frame.shape[1]  # Asumimos que z está en el mismo rango que x
                points.append((x, y, z))

            # Verificar el número de puntos faciales
            if len(points) != 478:
                print(f"Número de puntos faciales detectados: {len(points)}")
                continue  # Saltar si no se detectan 478 puntos faciales

            # Preprocesar los puntos faciales para el modelo
            points_array = np.array(points).flatten()
            points_normalized = (points_array - np.array([frame.shape[1] / 2, frame.shape[0] / 2, frame.shape[1] / 2] * (len(points_array) // 3))) / np.array([frame.shape[1] / 2, frame.shape[0] / 2, frame.shape[1] / 2] * (len(points_array) // 3))
            points_input = np.expand_dims(points_normalized, axis=0).astype(np.float32)

            # Verificar la forma del tensor de entrada
            if points_input.shape[1] != model.input_shape[1]:
                print(f"Dimension mismatch: expected {model.input_shape[1]}, got {points_input.shape[1]}")
                continue  # Saltar si la forma no coincide

            # Realizar la inferencia
            output_data = model.predict(points_input)

            # Imprimir los resultados de la inferencia
            print("Resultados de la inferencia:", output_data)

            # Obtener la predicción
            pred = np.argmax(output_data)

            # Definir la etiqueta y el color del rectángulo
            label = 'Estable' if pred == 0 else 'Inestable'
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)

            # Dibujar el rectángulo y la etiqueta en el cuadro
            x_min = int(min([p[0] for p in points]))
            y_min = int(min([p[1] for p in points]))
            x_max = int(max([p[0] for p in points]))
            y_max = int(max([p[1] for p in points]))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    else:
        print("No se detectaron puntos faciales.")

    # Mostrar el cuadro con las detecciones
    cv2.imshow('Detección de Emociones', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()