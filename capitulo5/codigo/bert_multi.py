# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:37:33 2024

@author: PC
"""

# A dependency of the preprocessing for BERT inputs
#pip install -U "tensorflow-text==2.13.*"
#pip install "tf-models-official==2.13.*"
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import gdown
import pandas as pd
import numpy as np
from official.nlp import optimization  # to create AdamW optimizer
from google.colab import drive
tf.get_logger().setLevel('ERROR')

def cargar_dataset(archivo_id,nom_salida):
  archivo_id = archivo_id
  url = f'https://drive.google.com/uc?id={archivo_id}'
  salida = nom_salida
  gdown.download(url, salida, quiet=False)
  df = pd.read_excel(nom_salida)
  df['comentario'] = df['comentario'].astype(str)

  return df

def crear_datos_tensor(df,lote_tam, seed, train_tam,val_tam):
  AUTOTUNE = tf.data.AUTOTUNE
  lote_tam = lote_tam
  seed = seed

# Dividir el dataset en conjunto de entrenamiento, validación y prueba
  train_tam = int(train_tam * len(df))
  val_tam = int(val_tam * len(df))
  test_tam = len(df) - train_tam - val_tam

  train_ds = tf.data.Dataset.from_tensor_slices((df['comentario'][:train_tam], df['label'][:train_tam]))
  val_ds = tf.data.Dataset.from_tensor_slices((df['comentario'][train_tam:train_tam+val_tam], df['label'][train_tam:train_tam+val_tam]))
  test_ds = tf.data.Dataset.from_tensor_slices((df['comentario'][-test_tam:], df['label'][-test_tam:]))

# Preparar los conjuntos de datos para el entrenamiento
  train_ds = train_ds.shuffle(buffer_size=train_tam, seed=seed).batch(lote_tam).cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.batch(lote_tam).cache().prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.batch(lote_tam).cache().prefetch(buffer_size=AUTOTUNE)

  return train_ds, val_ds, test_ds

def seleccionar_modelo_bert(bert_model_nom):
    # Diccionarios que contienen las URL de los modelos BERT y sus preprocesamientos
    mapa_bert = {
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1'
}
    mapa_preprocesar = {
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
}
    # Obtener las URL correspondientes según el nombre del modelo especificado
    nombre_bert = mapa_bert[bert_model_nom]
    nombre_preprocesador = mapa_preprocesar[bert_model_nom]
    # Retornar las URL
    return nombre_bert, nombre_preprocesador

def modelo_clasificador(nombre_modelo,nombre_preprocesador):
  texto_entrada = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  capa_preprocesamiento = hub.KerasLayer(nombre_preprocesador, name='preprocessing')
  entradas_cod = capa_preprocesamiento(texto_entrada)
  codificador = hub.KerasLayer(nombre_modelo, trainable=True, name='BERT_encoder')
  salidas = codificador(entradas_cod)
  net = salidas['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(3, activation='softmax', name='classifier')(net)
  return tf.keras.Model(texto_entrada, net)

def crear_perdida_metrica(logits):
    """Crea y devuelve una función de pérdida y una métrica para un modelo de clasificación.
    Args:
    logits (bool): Si es True, la función de pérdida tratará los resultados como logits.
                       Si es False, tratará los resultados como probabilidades. Por defecto es False.
"""
    # Definir la función de pérdida (SparseCategoricalCrossentropy)
    perdida = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=logits)
    # Definir la métrica (SparseCategoricalAccuracy)
    metrica = tf.metrics.SparseCategoricalAccuracy()

    return perdida, metrica

def crear_optimizador(entrenamiento_dt, epocas,cal_relacion,ini_lr,optimizador):
    # Calcular el número de pasos por época y el número total de pasos de entrenamiento
    pasos_por_epoca = tf.data.experimental.cardinality(entrenamiento_dt).numpy()
    num_pasos_entrenamiento = pasos_por_epoca * epocas
    # Calcular el número de pasos de calentamiento
    num_pasos_cal = int(cal_relacion * num_pasos_entrenamiento)
    # Crear el optimizador
    optimizador1 = optimization.create_optimizer(init_lr=ini_lr,
                                              num_train_steps=num_pasos_entrenamiento,
                                              num_warmup_steps=num_pasos_cal,
                                              optimizer_type=optimizador)

    return optimizador1

def compilar_modelo_clasificador(modelo_clasificador, optimizador, perdida, metrica):

    modelo_clasificador.compile(
        optimizer=optimizador,
        loss=perdida,
        metrics=metrica
    )
    print(f"El modelo ha sido compilado con el optimizador {optimizador}, la función de pérdida {perdida} y las métricas {metrica}.")

def entrenar_modelo(tipo_modelo,modelo_clasificador, entrenamiento_dt, val_dt,epocas):
  print(f'Training model with {tipo_modelo}')
  historia = modelo_clasificador.fit(x=entrenamiento_dt,
                               validation_data=val_dt,
                               epochs=epocas)
  
def evaluar_modelo(modelo_clasificador, test_dt):
    perdida, precision = modelo_clasificador.evaluate(test_dt)
    print(f'Loss: {perdida}')
    print(f'Accuracy: {precision}')
    return perdida, precision

def obtener_predicciones(modelo_clasificador,dt,nombre,ruta_dt_coment):
  # Convierte un conjunto de datos pandas en un arreglo numpy para la prediccion del modelo
  dt.columns = ['comentario']
  dt['comentario'] = dt['comentario'].astype(str)
  dt_numpy = dt.values
  # Obtener las predicciones del modelo para el conjunto definido
  etiquetas_predichas = modelo_clasificador.predict(dt_numpy)
  print(etiquetas_predichas)
  # Convertir las predicciones en etiquetas predichas
  etiquetas = np.argmax(etiquetas_predichas, axis=1)
  # Mostrar las etiquetas predichas
  print("Etiquetas predichas:", etiquetas)
  num_elementos = len(etiquetas)
  print("Número de elementos en 'etiquetas':", num_elementos)
  df_etiqueta = pd.DataFrame(etiquetas)
  # Supongamos que df es tu DataFrame aniaimos el nombre de la columna con predicciones
  df_etiqueta.columns = [nombre]
  # conjunto de datos a concatenar con los resultados
  drive.mount('/content/drive')
  dt = pd.read_csv(ruta_dt_coment)
  print("numero filas de conjunto de datos",dt.shape[0])
  # concatenando los comentarios y el resultado
  resultado = pd.concat([dt, df_etiqueta], axis=1)
  return resultado

def almacenar_resultado(dt,nombre):
  drive.mount('/content/drive')
# Supongamos que tu DataFrame de pandas se llama dataset se guardara como 'dataset.csv'
  dt.to_csv('/content/drive/MyDrive/'+nombre+'.csv', index=False)
    
def guardar_modelo_drive(modelo_clasificador,nombre,incluye_optimizador):
  modelo_clasificador.save('/content/drive/MyDrive/'+nombre, include_optimizer=incluye_optimizador)