import os
import shutil

def dividir_carpeta(carpeta_origen, tamano_maximo_mb=100):
    """
    Divide los archivos de una carpeta en subcarpetas de máximo 100 MB.
    
    :param carpeta_origen: Ruta de la carpeta original a dividir
    :param tamano_maximo_mb: Tamaño máximo de cada subcarpeta en MB
    """
    # Verificar que la carpeta de origen existe
    if not os.path.exists(carpeta_origen):
        print(f"La carpeta {carpeta_origen} no existe.")
        return

    # Obtener todos los archivos en la carpeta de origen
    archivos = [os.path.join(carpeta_origen, f) for f in os.listdir(carpeta_origen) 
                if os.path.isfile(os.path.join(carpeta_origen, f))]
    
    # Ordenar archivos por tamaño (opcional, pero puede ayudar en la distribución)
    archivos.sort(key=os.path.getsize)
    
    # Preparar variables para la división
    subcarpeta_actual = None
    subcarpeta_numero = 1
    tamano_actual_subcarpeta = 0

    # Crear la primera subcarpeta
    nombre_subcarpeta = os.path.join(carpeta_origen, f"Subcarpeta_{subcarpeta_numero}")
    os.makedirs(nombre_subcarpeta, exist_ok=True)

    # Iterar sobre los archivos
    for archivo in archivos:
        tamano_archivo = os.path.getsize(archivo)
        tamano_archivo_mb = tamano_archivo / (1024 * 1024)

        # Si el archivo supera el tamaño máximo, crear una nueva subcarpeta
        if tamano_archivo_mb > tamano_maximo_mb:
            print(f"Advertencia: El archivo {os.path.basename(archivo)} es más grande que 100 MB")
        
        # Verificar si necesitamos crear una nueva subcarpeta
        if tamano_actual_subcarpeta + tamano_archivo_mb > tamano_maximo_mb:
            subcarpeta_numero += 1
            nombre_subcarpeta = os.path.join(carpeta_origen, f"Subcarpeta_{subcarpeta_numero}")
            os.makedirs(nombre_subcarpeta, exist_ok=True)
            tamano_actual_subcarpeta = 0

        # Copiar el archivo a la subcarpeta actual
        ruta_destino = os.path.join(nombre_subcarpeta, os.path.basename(archivo))
        shutil.copy2(archivo, ruta_destino)
        
        # Actualizar el tamaño actual de la subcarpeta
        tamano_actual_subcarpeta += tamano_archivo_mb

    print(f"Archivos divididos en {subcarpeta_numero} subcarpetas.")

# Ejemplo de uso
if __name__ == "__main__":
    carpeta_a_dividir = r"C:\Users\carolina\Documents\Proyectos_programacion\Reconocimiento_emociones_modelo\Puntos_faciales\JSON_de_imagenes\Entrenamiento\Inestable"
    dividir_carpeta(carpeta_a_dividir)