import pandas as pd
import csv
from datetime import datetime

# Leer el archivo "Resultados_temp.csv" y asignarlo a una variable llamada "resultados_temp"
resultados_temp = pd.read_csv('Procesamiento_datos/Datos/Resultados_temp.csv')

# Almacenar la fecha, hora de inicio (del primer registro) y la hora de fin (del último registro) en variables
fecha = resultados_temp['fecha'].iloc[0]
hora_inicio = resultados_temp['hora'].iloc[0]
hora_fin = resultados_temp['hora'].iloc[-1]

# Identificar el último valor de la columna 'id_res_sesion' en el archivo "Resultado_sesion.csv" y asignarle el siguiente valor
try:
    resultado_sesion = pd.read_csv('Procesamiento_datos/Datos/Resultado_sesion.csv')
    id_res_sesion = resultado_sesion['id_res_sesion'].max() + 1 if not resultado_sesion.empty else 0
except FileNotFoundError:
    id_res_sesion = 0

id_paciente = 0  # ID del paciente (asumido)

# Guardar la fecha, hora de inicio y hora de fin en un archivo CSV llamado "Resultado_sesion.csv"
nuevo_resultado_sesion = pd.DataFrame({
    'id_res_sesion': [id_res_sesion],
    'id_paciente': [id_paciente],
    'fecha': [fecha],
    'hora_inicio': [hora_inicio],
    'hora_fin': [hora_fin]
})
nuevo_resultado_sesion.to_csv('Procesamiento_datos/Datos/Resultado_sesion.csv', mode='a', index=False, header=not 'resultado_sesion' in locals())

# Leer el último id_lote del archivo Lote_fotogramas.csv si existe
try:
    df_lote_existente = pd.read_csv('Procesamiento_datos/Datos/Lote_fotogramas.csv')
    ultimo_id_lote = df_lote_existente['id_lote'].max() if not df_lote_existente.empty else -1
except FileNotFoundError:
    ultimo_id_lote = -1

# Determinar el número de fotogramas procesados, el valor mínimo, máximo y promedio de cada bloque de 1 minuto
resultados_temp['hora'] = pd.to_datetime(resultados_temp['hora'], format='%H:%M:%S')
hora_inicio_sesion = resultados_temp['hora'].min()
resultados_temp['minutos_desde_inicio'] = (resultados_temp['hora'] - hora_inicio_sesion).dt.total_seconds() / 60
resultados_temp['bloque'] = resultados_temp['minutos_desde_inicio'].astype(int)

bloques = resultados_temp.groupby('bloque')

# Leer el archivo Estado_sm.csv una sola vez
estados_sm = pd.read_csv('Procesamiento_datos/Datos/Estado_sm.csv')

# Crear un DataFrame para almacenar los resultados de cada bloque
lote_fotogramas = pd.DataFrame(columns=['id_lote', 'n_fotogramas', 'valor_min_pred', 'valor_max_pred', 
                                        'valor_prom_pred', 'id_estado_sm', 'fecha', 'hora_inicio', 'hora_fin', 
                                        'prob_error_pred', 'id_res_sesion'])

# Iterar sobre cada bloque y calcular los valores requeridos
for bloque, datos in bloques:
    n_fotogramas_bloque = len(datos)
    valor_min_bloque = datos['valor_prediccion'].min()
    valor_max_bloque = datos['valor_prediccion'].max()
    valor_promedio_bloque = datos['valor_prediccion'].mean()

    hora_inicio_bloque = datos['hora'].min().strftime('%H:%M:%S')
    hora_fin_bloque = datos['hora'].max().strftime('%H:%M:%S')
    fecha_bloque = datos['fecha'].iloc[0]

    # Encontrar el estado correspondiente al valor promedio
    id_estado_sm_bloque = estados_sm[(estados_sm['valor_min'] <= valor_promedio_bloque) & 
                                     (valor_promedio_bloque < estados_sm['valor_max'])]['id_estado_sm'].iloc[0]

    # Contar el número de fotogramas en cada estado
    n_estados = {estado: len(datos[(datos['valor_prediccion'] >= estados_sm.loc[estados_sm['id_estado_sm'] == estado, 'valor_min'].iloc[0]) & 
                                   (datos['valor_prediccion'] < estados_sm.loc[estados_sm['id_estado_sm'] == estado, 'valor_max'].iloc[0])]) 
                 for estado in estados_sm['id_estado_sm']}

    # Estado con mayor cantidad de registros
    id_estado_mayor = max(n_estados, key=n_estados.get)
    n_estado_mayor = n_estados[id_estado_mayor]

    # Calcular la probabilidad de error de la predicción
    prob_error_pred_bloque = (n_fotogramas_bloque - n_estado_mayor) / n_fotogramas_bloque

    # Guardar los resultados en el DataFrame
    nuevo_lote = pd.DataFrame({
        'id_lote': [ultimo_id_lote + bloque + 1],
        'n_fotogramas': [n_fotogramas_bloque],
        'valor_min_pred': [valor_min_bloque],
        'valor_max_pred': [valor_max_bloque],
        'valor_prom_pred': [valor_promedio_bloque],
        'id_estado_sm': [id_estado_mayor],
        'fecha': [fecha_bloque],
        'hora_inicio': [hora_inicio_bloque],
        'hora_fin': [hora_fin_bloque],
        'prob_error_pred': [prob_error_pred_bloque],
        'id_res_sesion': [id_res_sesion]
    })
    
    lote_fotogramas = pd.concat([lote_fotogramas, nuevo_lote], ignore_index=True)

# Guardar los resultados en un archivo CSV llamado "Lote_fotogramas.csv"
lote_fotogramas.to_csv('Procesamiento_datos/Datos/Lote_fotogramas.csv', mode='a', index=False, 
                       header=not ultimo_id_lote >= 0, float_format='%.3f')

# Calcular el porcentaje de frecuencia de cada estado en la sesión completa
from datetime import datetime

# Convertir las cadenas de hora a objetos datetime
hora_inicio_dt = datetime.strptime(hora_inicio, '%H:%M:%S')
hora_fin_dt = datetime.strptime(hora_fin, '%H:%M:%S')

# Calcular la duración de la sesión en minutos
duracion_sesion = (hora_fin_dt - hora_inicio_dt).total_seconds() / 60

# Leer el archivo "Lote_fotogramas.csv" y asignarlo a una variable llamada "lote_fotogramas"
lote_fotogramas = pd.read_csv('Procesamiento_datos/Datos/Lote_fotogramas.csv')

# Filtrar los bloques que pertenecen a la sesión actual
lote_fotogramas_sesion = lote_fotogramas[lote_fotogramas['id_res_sesion'] == id_res_sesion]

# Calcular la duración de cada registro en segundos
lote_fotogramas_sesion['duracion'] = lote_fotogramas_sesion.apply(
    lambda row: (datetime.strptime(row['hora_fin'], '%H:%M:%S') - datetime.strptime(row['hora_inicio'], '%H:%M:%S')).total_seconds(),
    axis=1
)

# Calcular el porcentaje de frecuencia de cada estado en base a la duración de cada registro
frecuencia_estados = {}
for estado in estados_sm['id_estado_sm']:
    duracion_estado = lote_fotogramas_sesion[lote_fotogramas_sesion['id_estado_sm'] == estado]['duracion'].sum()
    frecuencia_estados[estado] = duracion_estado / (duracion_sesion * 60)  # Convertir duracion_sesion a segundos



# Calcular el porcentaje de frecuencia de cada estado en base a los estados de cada 
# lote de fotogramas (id_estado_sm) y duración que representa (hora_fin - hora_inicio)
# en minutos
'''frecuencia_estados = {estado: len(lote_fotogramas_sesion[lote_fotogramas_sesion['id_estado_sm'] == estado]) / duracion_sesion
                      for estado in estados_sm['id_estado_sm']}'''

# Guardar los resultados en un DataFrame
frecuencia_estados_df = pd.DataFrame({
    'id_res_sesion': [id_res_sesion] * len(frecuencia_estados),
    'id_estado_sm': list(frecuencia_estados.keys()),
    'por_frecuencia': list(frecuencia_estados.values())
})

# Guardar los resultados en un archivo CSV llamado "Detalle_resultado2.csv"
frecuencia_estados_df.to_csv('Procesamiento_datos/Datos/Detalle_resultado2.csv', mode='a', index=False, 
                             header=not pd.io.common.file_exists('Procesamiento_datos/Datos/Detalle_resultado2.csv'), 
                             float_format='%.2f')

print("Procesamiento completado con éxito.")