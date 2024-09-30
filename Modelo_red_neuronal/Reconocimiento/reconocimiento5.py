# Descripción: Código para realizar el reconocimiento de emociones en tiempo real utilizando un modelo TFLite.
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path=r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Modelos\model_estable_inestable.tflite')
interpreter.allocate_tensors()

# Obtener los detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Cargar el clasificador de Haar Cascades para detectar rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # Mostrar el frame procesado
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona Escape para cerrar
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
