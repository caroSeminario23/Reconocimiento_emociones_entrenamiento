# Descripción: Este script realiza la inferencia en tiempo real de un modelo TFLite para detectar si una persona está estable o inestable emocionalmente en un video.
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path=r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Modelos\model_estable_inestable.tflite')
interpreter.allocate_tensors()

# Obtener los detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Iniciar la captura de video desde un archivo
cap = cv2.VideoCapture(r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Videos\Ansiedad\video20.mp4')  # Cambia esto por la ruta de tu video

# Obtener el tamaño del video
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Cargar el clasificador de Haar Cascades para detectar rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Obtener la tasa de cuadros por segundo (FPS)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Obtener el tamaño de la pantalla
screen_width = 1920  # Cambia esto por la resolución de tu pantalla
screen_height = 1080  # Cambia esto por la resolución de tu pantalla

# Calcular la relación de aspecto
aspect_ratio = video_width / video_height

# Determinar el tamaño de la ventana
if video_width > screen_width or video_height > screen_height:
    if aspect_ratio > 1:  # Ancho mayor que alto
        window_width = screen_width
        window_height = int(screen_width / aspect_ratio)
    else:  # Alto mayor que ancho
        window_height = screen_height
        window_width = int(screen_height * aspect_ratio)
else:
    window_width = video_width
    window_height = video_height

# Crear una ventana y redimensionarla
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # Crear ventana ajustable
cv2.resizeWindow('Video', window_width, window_height)  # Redimensionar según el video

# Reproducir el video frame por frame y realizar la inferencia
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Salir si no hay más frames

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extraer la región del rostro
        face = frame[y:y+h, x:x+w]

        # Preprocesar el rostro
        img = cv2.resize(face, (48, 48))  # Redimensionar a 48x48
        img = img.astype(np.float32) / 255.0  # Normalizar
        img = np.expand_dims(img, axis=0)  # Añadir dimensión del batch

        # Realizar la inferencia
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Procesar la salida
        estado = "Inestable" if output_data[0][0] > 0.5 else "Estable"

        # Dibujar un rectángulo alrededor del rostro y mostrar el estado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, estado, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Redimensionar el frame para mantener la relación de aspecto
    frame_resized = cv2.resize(frame, (window_width, window_height))
    # Mostrar el frame procesado
    cv2.imshow('Video', frame_resized)

    # Esperar según la tasa de cuadros del video
    key = cv2.waitKey(int(1000 / fps))
    if key & 0xFF == 27:  # Presiona Escape para cerrar
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
