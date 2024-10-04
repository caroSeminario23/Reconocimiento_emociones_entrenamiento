# Leer el archivo "Resultados_temp.csv" y asignarlo a una variable llamada "resultados_temporales"
import csv
from datetime import datetime
import pandas as pd


resultados_temporales = pd.read_csv('Procesamiento_datos/Datos/Resultados_temp.csv')

# Almacenar la fecha, hora de inicio (del primer registro) y la hora de fin (del último registro) en variables
fecha = resultados_temporales['fecha'][0]
hora_inicio = resultados_temporales['hora'][0]
hora_fin = resultados_temporales['hora'][len(resultados_temporales)-1]

# Identificar el último valor de la columna 'id_res_sesion' en el archivo "Resultado_sesion.csv" y asignarle el siguiente valor
try:
    with open('Procesamiento_datos/Datos/Resultado_sesion.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        ids = [int(row['id_res_sesion']) for row in reader]
        id_res_sesion = max(ids) + 1 if ids else 0
except FileNotFoundError:
    id_res_sesion = 0

id_paciente = 1  # ID del paciente (asumido)

# Guardar la fecha, hora de inicio y hora de fin en un archivo CSV llamado "Resultado_sesion.csv"
with open('Procesamiento_datos/Datos/Resultado_sesion.csv', 'a', newline='') as csvfile:  # Cambiar a modo adición ('a')
    writer = csv.writer(csvfile)

    # Escribir los encabezados solo si el archivo está vacío
    if csvfile.tell() == 0:
        writer.writerow(['id_res_sesion', 'id_paciente', 'fecha', 'hora_inicio', 'hora_fin'])

    # Escribir los datos en el archivo CSV
    writer.writerow([id_res_sesion, id_paciente, fecha, hora_inicio, hora_fin])

# Determinar el número de fotogramas procesados, el valor mínimo, máximo y promedio de cada bloque de 1 minuto
# Dividir los registros en bloques de 1 minuto
resultados_temporales['hora'] = pd.to_datetime(resultados_temporales['hora']) # Convertir la columna 'hora' a tipo datetime
resultados_temporales['minuto'] = resultados_temporales['hora'].dt.minute # Extraer el minuto de la hora
resultados_temporales['bloque'] = resultados_temporales['hora'].dt.hour * 60 + resultados_temporales['minuto'] # Calcular el bloque de 1 minuto
bloques = resultados_temporales.groupby('bloque') # Agrupar los registros por bloque

# Crear un DataFrame para almacenar los resultados de cada bloque
resultados_por_minuto = pd.DataFrame(columns=['id_lote', 'n_fotogramas', 'valor_min_pred', 'valor_max_pred', 
                                              'valor_prom_pred', 'id_estado_sm', 'fecha', 'hora_inicio', 'hora_fin', 
                                              'prob_error_pred', 'id_res_sesion'])

# Iterar sobre cada bloque y calcular los valores requeridos
for bloque, datos in bloques:
    n_fotogramas = len(datos)
    valor_min = datos['valor_prediccion'].min()
    valor_max = datos['valor_prediccion'].max()
    valor_promedio = datos['valor_prediccion'].mean()

    # Calcular la hora de inicio y fin del bloque
    hora_inicio_bloque = datos['hora'].min().strftime('%H:%M:%S')
    hora_fin_bloque = datos['hora'].max().strftime('%H:%M:%S')

    # Contrastar el valor promedio con los umbrales de inestabilidad que están en el archivo Estado_sm.csv y guardar el id_estado_sm
    # del estado correspondiente (el que comprenda dentro de sus límites al valor promedio)
    estados_sm = pd.read_csv('Procesamiento_datos/Datos/Estado_sm.csv')

    # Verificar las columnas del DataFrame
    print("Columnas de estados_sm:", estados_sm.columns)

    id_estado_sm = None
    for i, estado in estados_sm.iterrows():
        if estado['valor_min'] <= valor_promedio < estado['valor_max']:
            id_estado_sm = estado['id_estado_sm']
            break
    
    # Calcular el número de registros por cada estado (almacenado en el archivo "Estado_sm.csv") en el bloque
    n_estados = datos['estado_detallado'].value_counts()

    # Calcular la probabilidad de error de la prediccion en base a la diferencia entre el número de registros con el estado que menos
    # se repite y el número total de registros del bloque 
    prob_error_pred = 1 - n_estados.max() / n_fotogramas


    # Guardar los resultados en el DataFrame
    resultados_por_minuto = resultados_por_minuto.append({'id_lote': bloque, 'n_fotogramas': n_fotogramas, 
                                                          'valor_min_pred': valor_min, 'valor_max_pred': valor_max, 
                                                          'valor_prom_pred': valor_promedio, 'id_estado_sm': id_paciente,
                                                          'fecha': fecha, 'hora_inicio': hora_inicio_bloque, 'hora_fin': hora_fin_bloque,
                                                          'prob_error_pred': prob_error_pred, 'id_res_sesion': id_res_sesion}, 
                                                          ignore_index=True)

# Guardar los resultados en un archivo CSV llamado "Lote_fotogramas.csv"
with open('Procesamiento_datos/Datos/Lote_fotogramas.csv', 'a', newline='') as csvfile: 
    writer = csv.writer(csvfile)

    # Escribir los encabezados solo si el archivo está vacío
    if csvfile.tell() == 0:
        writer.writerow(['id_lote', 'n_fotogramas', 'valor_min_pred', 'valor_max_pred', 
                         'valor_prom_pred', 'id_estado_sm', 'fecha', 'hora_inicio', 'hora_fin', 
                         'prob_error_pred', 'id_res_sesion'])

    # Escribir los datos en el archivo CSV
    for _, row in resultados_por_minuto.iterrows():
        writer.writerow(row)

# Guardar los resultados en un archivo CSV llamado "Lote_fotogramas.csv"
#resultados_por_minuto.to_csv('Procesamiento_datos/Datos/Lote_fotogramas.csv', index=False)

# Calcular el porcentaje de frecuencia de cada estado en la sesión en base a los estados guardados en el archivo "Estado_sm.csv"
estados_sm = pd.read_csv('Procesamiento_datos/Datos/Estado_sm.csv')
frecuencia_estados = resultados_temporales['estado_detallado'].value_counts(normalize=True) * 100
frecuencia_estados = frecuencia_estados.reindex(estados_sm['estado_sm']).fillna(0)

# Guardar los resultados en un archivo CSV llamado "Detalle_resultado.csv" con los campos 'id_res_sesion', 'id_estado_sm', 'por_frecuencia'
with open('Procesamiento_datos/Datos/Detalle_resultado.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Escribir los encabezados solo si el archivo está vacío
    if csvfile.tell() == 0:
        writer.writerow(['id_res_sesion', 'id_estado_sm', 'por_frecuencia'])

    # Escribir los datos en el archivo CSV
    for id_estado_sm, por_frecuencia in frecuencia_estados.items():
        writer.writerow([id_res_sesion, id_estado_sm, por_frecuencia])