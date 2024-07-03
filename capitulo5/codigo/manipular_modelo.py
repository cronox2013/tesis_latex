# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:48:32 2024

@author: PC
"""
import pickle 
import keras
#pip install keras
#pip install tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import cnn_dos, preparar_datos

def cargar_tokenizer(ruta_tokenizer):
    # Cargar el Tokenizer desde el archivo
    with open(ruta_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer
    
def preprocesamiento_entrada(tokenizer,max_tam_secuencia, texto_entrada):
    nuevas_secuencias = tokenizer.texts_to_sequences([texto_entrada])
    # Pad las secuencias a la longitud esperada por el modelo
    texto_pad = pad_sequences(nuevas_secuencias, maxlen=max_tam_secuencia)
    # Verificar la forma de la secuencia preparada
    print(texto_pad.shape)
    return texto_pad

def cargar_modelo(ruta_matriz,ruta_modelo):
    embedding_matriz = np.load(ruta_matriz)
    print("Matriz de embeddings cargada correctamente.")
    print(f"Forma de la matriz de embeddings cargada: {embedding_matriz.shape}")
    # Crear una nueva capa de embedding utilizando la matriz cargada
    embedding_layer = preparar_datos.capa_embedding(30270, 300, embedding_matriz, 150, False)
    # Crear la estructura del mejor modelo
    cnn = cnn_dos.cnn_dp_two(embedding_layer)
    # Cargar los pesos al mejor modelo
    cnn.load_weights(ruta_modelo)  # Usar la misma ruta y extensión
    cnn.summary()
    return cnn

def predecir(modelo,entrada):
    predicciones = modelo.predict(entrada)
    print(predicciones)
    return predicciones
 
def prediccion_cnn(ruta_tokenizer,texto_entrada,ruta_matriz,ruta_modelo):
    tokenizer = cargar_tokenizer(ruta_tokenizer)
    texto_pad=preprocesamiento_entrada(tokenizer, 150, texto_entrada)
    modelo = cargar_modelo(ruta_matriz,ruta_modelo)
    resultado = predecir(modelo, texto_pad)
    print(resultado)
    return resultado