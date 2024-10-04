import cv2 # Para procesamiento de imágenes y vídeos
import mediapipe as mp # Para procesamiento de malla facial
import json # Para manejar archivos JSON
import os # Para trabajar con archivos y directorios

mp_drawing = mp.solutions.drawing_utils # Utilidades para dibujar en la imagen
mp_drawing_styles = mp.solutions.drawing_styles # Estilos de dibujo predefinidos
mp_face_mesh = mp.solutions.face_mesh # Módulo de malla facial

# Directorio de imágenes a procesar
image_dir = r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Imagenes\Fotogramas\Entrenamiento\Inestable'
output_image_dir = r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Puntos_faciales\Imagenes_con_malla\Entrenamiento\Inestable'
output_json_dir = r'C:\Users\carolina\Documents\VS Code\Reconocimiento_emociones_modelo\Puntos_faciales\JSON_de_imagenes\Entrenamiento\Inestable'

# Crear directorios de salida si no existen
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

# Listar todos los archivos en el directorio de imágenes
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) # Especificaciones para dibujar en la imagen: grosor y radio del círculo

# Inicia la malla facial con parámetros específicos.
with mp_face_mesh.FaceMesh( 
    static_image_mode=True, # Modo de imagen estática
    max_num_faces=1, # Número máximo de rostros a detectar
    refine_landmarks=True, # Refinar los puntos de referencia para mayor precisión
    min_detection_confidence=0.5) as face_mesh: # Confianza mínima de detección (umbral para considerar que ha detectado un rostro)
  
  for idx, file_name in enumerate(image_files): # Para cada archivo de imagen
    file_path = os.path.join(image_dir, file_name)
    image = cv2.imread(file_path) # Cargar la imagen desde la ruta especificada

    if image is None:
        print(f"Advertencia: No se pudo cargar la imagen {file_name}.")
        continue # Si la imagen no se puede cargar, continuar con la siguiente

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convertir la imagen de BGR a RGB y procesarla con la malla facial

    if not results.multi_face_landmarks: # Si no se detectaron rostros
      print(f"Advertencia: No se detectaron rostros en la imagen {file_name}.")
      continue # Continuar con la siguiente imagen

    annotated_image = image.copy() # Crear un copia de la imagen original para dibujar los puntos faciales sobre ella
    face_landmarks_list = [] # Lista para almacenar los puntos faciales detectados para esta imagen

    for face_landmarks in results.multi_face_landmarks: # Para cada rostro detectado
      print(f"Procesando {file_name}: {len(face_landmarks.landmark)} puntos faciales detectados.")
      #print('face_landmarks:', face_landmarks) # Imprimir los puntos faciales detectados en consola (habilitar si se desea ver las coordenadas de los puntos faciales en consola)

      # Almacenar los puntos faciales en la lista
      landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in face_landmarks.landmark]
      face_landmarks_list.append(landmarks)

      mp_drawing.draw_landmarks( # Dibujar los puntos de referencia de la malla facial
          image=annotated_image, # Imagen de entrada
          landmark_list=face_landmarks, # Lista de puntos de referencia del rostro
          connections=mp_face_mesh.FACEMESH_TESSELATION, # Conexiones de la malla facial
          landmark_drawing_spec=None, # Especificaciones de dibujo de los puntos de referencia
          connection_drawing_spec=mp_drawing_styles # Estilo de dibujo de las conexiones
          .get_default_face_mesh_tesselation_style()) # Estilo de dibujo de la malla facial
      
      mp_drawing.draw_landmarks( # Dibujar los contornos de la malla facial
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS, # Conexiones de los contornos de la malla facial
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles # Estilo de dibujo de los contornos
          .get_default_face_mesh_contours_style()) # Estilo de dibujo de los contornos de la malla facial
      
      mp_drawing.draw_landmarks( # Dibujar los puntos de referencia de los ojos
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES, # Conexiones de los ojos
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles # Estilo de dibujo de los ojos
          .get_default_face_mesh_iris_connections_style()) # Estilo de dibujo de los ojos en la malla facial
      
    # Guardar la imagen anotada en el directorio de salida
    output_image_path = os.path.join(output_image_dir, file_name)
    cv2.imwrite(output_image_path, annotated_image)

    # Guardar los puntos faciales detectados en un archivo JSON
    output_json_path = os.path.join(output_json_dir, f'{os.path.splitext(file_name)[0]}.json')
    with open(output_json_path, 'w') as f:
      json.dump(face_landmarks_list, f, indent=4) # Guardar la lista de puntos faciales en el archivo JSON