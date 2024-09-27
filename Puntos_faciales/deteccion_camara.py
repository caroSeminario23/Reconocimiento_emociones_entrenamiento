import cv2 # Importa la biblioteca OpenCV para el procesamiento de imágenes y video.
import mediapipe as mp # Importa la biblioteca MediaPipe para el procesamiento de malla facial.

mp_drawing = mp.solutions.drawing_utils # Utilidades para dibujar en la imagen: incluye las funciones draw_landmarks y draw_connections
mp_drawing_styles = mp.solutions.drawing_styles # Estilos de dibujo predefinidos de MediaPipe: incluye los estilos de malla facial
mp_face_mesh = mp.solutions.face_mesh # Módulo de malla facial de MediaPipe: incluye la detección y seguimiento de la malla facial

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) # Especificaciones para dibujar en la imagen: grosor y radio del círculo
cap = cv2.VideoCapture(0) # Capturar video desde la cámara

# Inicia la malla facial con parámetros específicos.
with mp_face_mesh.FaceMesh( 
    max_num_faces=1, # Número máximo de rostros a detectar
    refine_landmarks=True, # Refinar los puntos de referencia para mayor precisión
    min_detection_confidence=0.5, # Confianza mínima de detección (umbral para considerar que ha detectado un rostro)
    min_tracking_confidence=0.5) as face_mesh: # Confianza mínima de seguimiento (umbral para considerar que ha seguido un rostro)


  while cap.isOpened(): # Mientras la cámara esté abierta
    success, image = cap.read() # Leer un frame de la cámara
    if not success: # Si no se pudo leer el frame
      print("Ignoring empty camera frame.") # Imprimir mensaje de error
      continue # Continuar con el siguiente frame

    # Para mejorar el rendimiento, opcionalmente marca la imagen como no editable
    image.flags.writeable = False # Marcar la imagen como no editable
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convertir la imagen de BGR a RGB para el procesamiento de MediaPipe
    results = face_mesh.process(image) # Procesar la imagen con la malla facial

    # Marca la imagen como editable nuevamente
    image.flags.writeable = True # Marcar la imagen como editable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convertir la imagen de RGB a BGR

    if results.multi_face_landmarks: # Si se detectaron rostros
      for face_landmarks in results.multi_face_landmarks: # Para cada rostro detectado

        # Imprimir la cantidad de puntos faciales detectados
        print(f'Cantidad de puntos faciales detectados: {len(face_landmarks.landmark)}')

        mp_drawing.draw_landmarks( # Dibujar los puntos de referencia de la malla
            image=image, # Imagen de entrada
            landmark_list=face_landmarks, # Lista de puntos de referencia del rostro
            connections=mp_face_mesh.FACEMESH_TESSELATION, # Conexiones de la malla facial
            landmark_drawing_spec=None, # Especificaciones de dibujo de los puntos de referencia
            connection_drawing_spec=mp_drawing_styles # Estilo de dibujo de las conexiones
            .get_default_face_mesh_tesselation_style()) # Estilo de dibujo de la malla facial
        
        mp_drawing.draw_landmarks( # Dibuja los contornos de la malla facial
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS, # Conexiones de los contornos de la malla facial
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style()) # Estilo de dibujo de los contornos de la malla facial
        
        mp_drawing.draw_landmarks( # Dibujar los puntos de referencia de los ojos
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES, # Conexiones de los ojos
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style()) # Estilo de dibujo de las conexiones de los ojos
        
    # Voltear la imagen horizontalmente para una vista de selfie
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1)) # Mostrar la imagen con la malla facial
    if cv2.waitKey(5) & 0xFF == 27: # Si se presiona la tecla Esc
      break # Salir del bucle

cap.release() # Liberar la cámara