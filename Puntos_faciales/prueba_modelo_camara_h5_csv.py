import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import csv
from datetime import datetime

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
mp_drawing_styles = mp.solutions.drawing_styles

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)  # 0 indica la cámara predeterminada

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

print("Cámara iniciada correctamente.")

# Obtener las dimensiones de la pantalla
screen_width = 1920  # Ajusta esto al ancho de tu pantalla
screen_height = 1080  # Ajusta esto al alto de tu pantalla

# Función para escribir resultados en CSV
def write_to_csv(filename, data):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Fecha y Hora', 'Estado', 'Valor de Predicción'])
        writer.writerow(data)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    # Voltear el frame horizontalmente para un efecto de "espejo"
    frame = cv2.flip(frame, 1)

    # Redimensionar el frame si es más grande que la pantalla
    if frame.shape[1] > screen_width or frame.shape[0] > screen_height:
        scale = min(screen_width / frame.shape[1], screen_height / frame.shape[0])
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

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
                points.append((landmark.x, landmark.y, landmark.z))

            # Verificar el número de puntos faciales
            if len(points) != 478:
                print(f"Número de puntos faciales detectados: {len(points)}")
                continue  # Saltar si no se detectan 478 puntos faciales

            # Preprocesar los puntos faciales para el modelo
            points_array = np.array(points).flatten()
            points_input = np.expand_dims(points_array, axis=0).astype(np.float32)

            # Verificar la forma del tensor de entrada
            if points_input.shape[1] != model.input_shape[1]:
                print(f"Dimension mismatch: expected {model.input_shape[1]}, got {points_input.shape[1]}")
                continue  # Saltar si la forma no coincide

            # Realizar la inferencia
            output_data = model.predict(points_input)

            # Obtener el valor de predicción
            pred_value = output_data[0][0]  # Asumiendo que el modelo devuelve un solo valor

            # Determinar la etiqueta basada en el valor de predicción
            if pred_value < 0.5:
                label = 'Estable'
                color = (0, 255, 0)  # Verde para estable
            else:
                label = 'Inestable'
                color = (0, 0, 255)  # Rojo para inestable

            # Dibujar el rectángulo y la etiqueta en el cuadro
            h, w, _ = frame.shape
            x_min, y_min = int(min([p[0] for p in points]) * w), int(min([p[1] for p in points]) * h)
            x_max, y_max = int(max([p[0] for p in points]) * w), int(max([p[1] for p in points]) * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{label} ({pred_value:.2f})", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Guardar resultados en CSV
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            write_to_csv('resultados_estado_realtime.csv', [current_time, label, pred_value])

    else:
        print("No se detectaron puntos faciales.")

    # Mostrar el frame con las detecciones
    cv2.imshow('Detección de Emociones en Tiempo Real', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()