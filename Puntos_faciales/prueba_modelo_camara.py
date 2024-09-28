import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path=r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Puntos_faciales\Modelos_pf\model_estable_inestable.tflite')
interpreter.allocate_tensors()

# Obtener los detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Imprimir las dimensiones esperadas del tensor de entrada
print("Dimensiones esperadas del tensor de entrada:", input_details[0]['shape'])

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)

# Obtener el tamaño del video
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Inicializar MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Crear una ventana para mostrar el video
cv2.namedWindow('Detección de Emociones', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detección de Emociones', video_width, video_height)  # Redimensionar según el video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el cuadro a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el cuadro con MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

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
                x = landmark.x * video_width
                y = landmark.y * video_height
                z = landmark.z * video_width  # Asumimos que z está en el mismo rango que x
                points.append((x, y, z))

            # Verificar el número de puntos faciales
            if len(points) != 478:
                print(f"Número de puntos faciales detectados: {len(points)}")
                continue  # Saltar si no se detectan 478 puntos faciales

            # Preprocesar los puntos faciales para el modelo
            points_array = np.array(points).flatten()
            points_normalized = (points_array - np.array([video_width / 2, video_height / 2, video_width / 2] * (len(points_array) // 3))) / np.array([video_width / 2, video_height / 2, video_width / 2] * (len(points_array) // 3))
            points_input = np.expand_dims(points_normalized, axis=0).astype(np.float32)

            # Verificar la forma del tensor de entrada
            if points_input.shape[1] != input_details[0]['shape'][1]:
                print(f"Dimension mismatch: expected {input_details[0]['shape'][1]}, got {points_input.shape[1]}")
                continue  # Saltar si la forma no coincide

            # Realizar la inferencia
            interpreter.set_tensor(input_details[0]['index'], points_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

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