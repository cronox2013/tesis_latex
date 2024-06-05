# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:41:36 2024

@author: PC
"""

#pip install wget
import os
import wget
import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import Constant

def descargar_embedding(url,nombre_archivo):
  # Descarga el archivo utilizando wget
  !wget $url -O $nombre_archivo
  # Descomprime el archivo si está comprimido (en este caso, .gz)
  !gunzip $nombre_archivo
  
def convertir_a_listas(x_entre,x_val, x_pru, y_entre, y_val, y_pru):
  entre_texto = x_entre['comentario'].tolist()
  pru_texto = x_pru['comentario'].tolist()
  val_texto = x_val['comentario'].tolist()
  entre_etiqueta = y_entre['label'].tolist()
  pru_etiqueta = y_pru['label'].tolist()
  val_etiqueta = y_val['label'].tolist()
  return entre_texto, pru_texto, val_texto, entre_etiqueta, pru_etiqueta, val_etiqueta

def tokenizar_texto(entre_texto, test_texto, val_texto,max_num_palabras):
  tokenizador = Tokenizer(num_words=max_num_palabras)
  tokenizador.fit_on_texts(entre_texto)
  entre_secuencias = tokenizador.texts_to_sequences(entre_texto) #Converting text to a vector of word indexes
  test_secuencias = tokenizador.texts_to_sequences(test_texto)
  val_secuencias = tokenizador.texts_to_sequences(val_texto)
  indice_palabras = tokenizer.indice_palabras
  print('Found %s unique tokens.' % len(indice_palabras))
  return entre_secuencias, test_sequencias, val_sequencias, indice_palabras

def padding(entre_secuencias,test_secuencias,val_secuencias,max_tam_secuencia):
  #Converting this to sequences to be fed into neural network. Max seq. len is 1000 as set earlier
  #initial padding of 0s, until vector is of size MAX_SEQUENCE_LENGTH
  x_entre = pad_sequences(entre_secuencias, maxlen=max_tam_secuencia)
  test_datos = pad_sequences(test_secuencias, maxlen=max_tam_secuencia)
  x_val = pad_sequences(val_secuencias, maxlen=max_tam_secuencia)

  return x_entre, test_datos, x_val

def one_hot_etiquetas(entre_etiquetas, test_etiquetas, val_etiquetas):
  y_entre = to_categorical(np.asarray(entre_etiquetas))
  test_etiquetas = to_categorical(np.asarray(test_etiquetas))
  y_val = to_categorical(np.asarray(val_etiquetas))
  return y_entre, test_etiquetas, y_val


def cargar_vectores(name):
    fin = io.open(name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    datos = {}
    for linea in fin:
        tokens = linea.rstrip().split(' ')
        palabra = tokens[0]
        coefs = np.asarray(tokens[1:], dtype='float32')
        datos[palabra] = coefs

    print('n =', n)
    print('d =', d)
    return datos

def embedding_matriz(embeddings_indice,max_num_palabras,embedding_dim,indice_palabras):
  num_palabras = min(max_num_palabras, len(indice_palabras) + 1
  embedding_matriz = np.zeros((num_palabras, embedding_dim))
  for palabra, i in indice_palabras.items():
      if i > max_num_palabras:
          continue
      embedding_vector = embeddings_indice.get(palabra)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
          embedding_matriz[i] = embedding_vector

  return num_palabras,embedding_matriz

def capa_embedding(num_palabras, embedding_dim,embedding_matriz,max_tam_secuencia,entrenable):
  # load these pre-trained word embeddings into an Embedding layer
  # note that we set trainable = False so as to keep the embeddings fixed
  capa_de_embedding = Embedding(num_palabras,
                              embedding_dim,
                              embeddings_initializer=Constant(embedding_matriz),
                              input_length=max_tam_secuencia,
                              trainable=entrenable)
  print("Preparing of embedding matrix is done")
  return capa_de_embedding

def compilar_entrenar_modelo(cnn_modelo,optimizador,metricas,x_entre,y_entre,x_val,y_val,tam_lote,epocas,ruta_modelo):
  #Compilar modelo
  cnn_modelo.compile(loss='categorical_crossentropy',
                     optimizer= optimizador,
                     metrics=metricas)

  control = ModelCheckpoint(
        filepath=ruta_modelo,  # Ruta del archivo para guardar el modelo
        monitor='val_loss',         # Métrica a monitorizar (en este caso, pérdida en validación)
        save_best_only=True,        # Guardar solo si la métrica mejora
        mode='min',                 # Queremos minimizar la pérdida
        verbose=1                   # Muestra mensajes sobre el proceso
    )
  # Entrenar el modelo
  historia = cnn_modelo.fit(x_entre, y_entre,
                 batch_size=tam_lote,
                 epochs=epocas,
                 validation_data=(x_val, y_val),
                 callbacks=[control])


  return cnn_modelo, historia

def evaluar_modelo(cnn,test_datos, test_etiquetas):
      puntaje, precision = cnn.evaluate(test_datos, test_etiquetas)
      print('Precisión de prueba con CNN versión 2:', precision)
      
def graficar_modelo(historia):
  plt.figure(figsize=(10, 5))
  plt.plot(historia.history['acc'], label='Exactitud del entrenamiento')
  plt.plot(historia.history['val_acc'], label='Exactitud de la validación')
  plt.title('Exactitud a lo largo de las épocas')
  plt.xlabel('Época')
  plt.ylabel('Exactitud')
  plt.legend(loc='lower right')
  plt.show()

  # Graficar el error (pérdida)
  plt.figure(figsize=(10, 5))
  plt.plot(historia.history['loss'], label='Pérdida del entrenamiento')
  plt.plot(historia.history['val_loss'], label='Pérdida de la validación')
  plt.title('Pérdida a lo largo de las épocas')
  plt.xlabel('Época')
  plt.ylabel('Pérdida')
  plt.legend(loc='upper right')
  plt.show()
