# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:41:19 2024

@author: PC
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from wordcloud import WordCloud

def cargar_dataset(ruta):
  drive.mount('/content/drive')
  ruta_destino = '/content/drive/MyDrive/'+ruta
  df = pd.read_csv(ruta_destino)
  df['comentario']= df['comentario'].astype(str)
  return df

def obtener_longitud_secuencias(column_nombre, df):
# Inicializar una lista para almacenar las longitudes de las secuencias
  longitudes = df[column_nombre].apply(lambda secuencia: len(secuencia.split()))
# Agregar las longitudes como una nueva columna en el DataFrame (opcional)
  df['longitud'] = longitudes
# Calcular estadísticas
  num_secuencias = len(longitudes)
  longitud_maxima = longitudes.max()
  longitud_minima = longitudes.min()
# Imprimir los resultados
  print(f"Total de secuencias: {num_secuencias}")
  print(f"Longitud máxima de las secuencias: {longitud_maxima}")
  print(f"Longitud mínima de las secuencias: {longitud_minima}")
# Crear un histograma con seaborn para visualizar la distribución de las longitudes
  sns.histplot(longitudes, bins=20, kde=True)
  plt.xlabel('Longitud de las secuencias')
  plt.ylabel('Frecuencia')
  plt.title('Distribución de longitudes de las secuencias')
  plt.show()

def vocabulario_unico(df, column_nombre):
  vocabulario_unico = set()
# Recorrer cada secuencia en la columna de texto
  for secuencia in df[column_nombre]:
    # Dividir la secuencia en palabras (tokens)
      palabras = secuencia.split()
    # Agregar las palabras al conjunto (se agregarán solo palabras únicas)
      vocabulario_unico.update(palabras)
# Contar el número de diferentes palabras
  num_palabras_diferentes = len(vocabulario_unico)
# Imprimir el resultado
  print(f"El número de diferentes palabras en el dataset es: {num_palabras_diferentes}")
  
def nube_palabras(columna_texto, df):
# Combinar todas las secuencias de texto en un solo string
  texto_completo = df[columna_texto].str.cat(sep=' ')
# Crear la nube de palabras
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)
# Mostrar la nube de palabras
  plt.figure(figsize=(10, 5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show()
  
def eliminar_long_cero(df):
    df_sin_cero = df[df['longitud'] != 0]
    return df_sin_cero

def construir_dataset_manual(ofensivo,no_ofensivo,grosero,nombre_ent,nombre_val,nombre_pru):
  df1si = ofensivo.iloc[:10250]
  df2si = ofensivo.iloc[10250:12375]
  df3si = ofensivo.iloc[12375:14955]

  df1no = no_ofensivo.iloc[:10250]
  df2no = no_ofensivo.iloc[10250:12375]
  df3no = no_ofensivo.iloc[12375:14947]

  df1gro = grosero.iloc[:4000]
  df2gro = grosero.iloc[4000:5000]
  df3gro = grosero.iloc[5000:5098]

  entrenamiento = pd.concat([df1si,df1no,df1gro],axis=0)
  entrenamiento.to_csv('/content/drive/MyDrive/'+nombre_ent+'.csv',index=False)
  validacion = pd.concat([df2si,df2no,df2gro],axis=0)
  validacion.to_csv('/content/drive/MyDrive/'+nombre_val+'.csv',index=False)
  prueba = pd.concat([df3si,df3no,df3gro], axis=0)
  prueba.to_csv('/content/drive/MyDrive/'+nombre_pru+'.csv',index=False)