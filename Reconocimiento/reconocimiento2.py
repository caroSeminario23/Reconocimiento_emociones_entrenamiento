import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path= r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Modelos\model_estable_inestable.tflite')
interpreter.allocate_tensors()

# Obtener los detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar el frame (ajusta según tu modelo)
    img = cv2.resize(frame, (48, 48))  # Cambia ancho y alto según tu modelo
    img = img.astype(np.float32) / 255.0  # Normaliza si es necesario
    img = np.expand_dims(img, axis=0)  # Añadir dimensión del batch

    # Realizar la inferencia
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Procesar la salida
    if output_data[0][0] > 0.5:  # Ajusta el umbral según tu caso
        estado = "Inestable"
    else:
        estado = "Estable"

    # Mostrar el resultado en el frame
    cv2.putText(frame, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona Escape para cerrar
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
