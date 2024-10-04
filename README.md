# RECONOCIMIENTO FACIAL
## Entrenamiento de los modelos
### Basado en:
OmesTutorials2020 - 7 RECONOCIMIENTO DE EMOCIONES: 
[Enlace al repositorio base](https://github.com/GabySol/OmesTutorials2020/tree/master)

### Modelos:
1. Eigenfaces
2. Fisherfaces
3. LBPH
4. CNN con Tensorflow Lite
5. CNN con Tensorflow Lite utilizando puntos faciales y Mediapipe

### Pasos para su ejecución:
1. Crea un entorno virtual.
2. Instala las dependencias del archivo `requirements.txt` empleando el comando `pip install -r requirements.txt`.

3. Ejecuta el archivo `captura_emociones.py`.
4. Ejecuta el archivo `entrenamiento.py`.
5. Ejecuta el archivo `reconocimiento.py`.

### Ejemplo de funcionamiento (Modelos 1, 2 y 3):
![Ejemplo de funcionamiento](Pruebas/Demostracion.gif)

### Ejemplo de funcionamiento (Modelo 4):
<div style="display: flex; flex-wrap: wrap;">
    <img src="Pruebas/Demostracion2.png" alt="Ejemplo de funcionamiento2" width="200"/>
    <img src="Pruebas/Demostracion3.png" alt="Ejemplo de funcionamiento3" width="200"/>
    <img src="Pruebas/Demostracion4.png" alt="Ejemplo de funcionamiento4" width="200"/>
    <img src="Pruebas/Demostracion5.png" alt="Ejemplo de funcionamiento5" width="200"/>
</div>

### Ejemplo de funcionamiento (Modelo 5):
<div style="display: flex; flex-wrap: wrap;">
    <img src="Pruebas/Demostracion6.png" alt="Ejemplo de funcionamiento6" width="200"/>
    <img src="Pruebas/Demostracion7.png" alt="Ejemplo de funcionamiento7" width="200"/>
    <img src="Pruebas/Demostracion8.png" alt="Ejemplo de funcionamiento8" width="200"/>
    <img src="Pruebas/Demostracion9.png" alt="Ejemplo de funcionamiento9" width="200"/>
</div>

## Jerarquización de estado inestable
<div style="display: flex; flex-wrap: wrap;">
    <img src="Pruebas/Demostracion10.png" alt="Ejemplo de funcionamiento10" width="200"/>
    <img src="Pruebas/Demostracion11.png" alt="Ejemplo de funcionamiento11" width="200"/>
    <img src="Pruebas/Demostracion12.png" alt="Ejemplo de funcionamiento12" width="200"/>
    <img src="Pruebas/Demostracion13.png" alt="Ejemplo de funcionamiento13" width="200"/>
</div>

## Modelo lógico de la BD para almacenar los resultados (SQLite)
![base de datos](Database/modelo_logico.png)

## Resultados de procesamiento de la data temporal (sesión de análisis)
*Se muestra 5 registros por tabla*
1. Archivo Resultados_temp.csv

    ![Tabla 1](Pruebas/Demostracion14.png)

2. Archivo Estado_sm.csv

    ![Tabla 2](Pruebas/Demostracion15.png)

3. Archivo Resultado_sesion.csv

    ![Tabla 3](Pruebas/Demostracion16.png)

4. Archivo Detalle_resultado.csv

    ![Tabla 4](Pruebas/Demostracion17.png)

5. Archivo Lote_fotogramas.csv

    ![Tabla 5](Pruebas/Demostracion18.png)