import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('Procesamiento_datos/Datos/Lote_fotogramas.csv')

# Seleccionar los primeros 5 registros
df_sample = df.head(5)

# Convertir los registros seleccionados a formato Markdown
markdown_table = df_sample.to_markdown(index=False)
print(markdown_table)