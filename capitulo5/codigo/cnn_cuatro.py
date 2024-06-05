# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:41:59 2024

@author: PC
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import MaxPooling1D

def cnn_four(capa_embedding):
  cnn = Sequential()
  cnn.add(capa_embedding)
  cnn.add(Conv1D(32, 3, activation='relu'))
  cnn.add(MaxPooling1D(2,1))
  cnn.add(Conv1D(64, 5, activation='relu'))
  cnn.add(MaxPooling1D(2,1))
  cnn.add(Conv1D(128, 5, activation='relu'))
  cnn.add(MaxPooling1D(2,1))
  cnn.add(Conv1D(256, 7, activation='relu'))
  cnn.add(GlobalMaxPooling1D())
  cnn.add(Dense(128, activation='relu'))
  cnn.add(Dense(3, activation='softmax'))

  return cnn

def cnn_dp_four(capa_embedding):
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa Conv1D con Dropout
    cnn.add(Conv1D(32, 3, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(MaxPooling1D(2,1))
    # Segunda capa Conv1D con Dropout
    cnn.add(Conv1D(64, 5, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(MaxPooling1D(2,1))
    # Tercera capa Conv1D con Dropout
    cnn.add(Conv1D(128, 5, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(MaxPooling1D(2,1))
    # Cuarta capa Conv1D con Dropout
    cnn.add(Conv1D(256, 7, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    # Capa de salida
    cnn.add(Dense(3, activation='softmax'))

    return cnn

def cnn_dp_four_f(capa_embedding):
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa Conv1D con Dropout
    cnn.add(Conv1D(32, 3, activation='relu'))
    cnn.add(Dropout(0.4))  # Añade Dropout con tasa del 40%
    cnn.add(MaxPooling1D(2,1))
    # Segunda capa Conv1D con Dropout
    cnn.add(Conv1D(64, 5, activation='relu'))
    cnn.add(Dropout(0.4))  # Añade Dropout con tasa del 40%
    cnn.add(MaxPooling1D(2,1))
    # Tercera capa Conv1D con Dropout
    cnn.add(Conv1D(128, 5, activation='relu'))
    cnn.add(Dropout(0.4))  # Añade Dropout con tasa del 40%
    cnn.add(MaxPooling1D(2,1))
    # Cuarta capa Conv1D con Dropout
    cnn.add(Conv1D(256, 7, activation='relu'))
    cnn.add(Dropout(0.4))  # Añade Dropout con tasa del 40%
    cnn.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    # Capa de salida
    cnn.add(Dense(3, activation='softmax'))

    return cnn

def cnn_dp_four_fi(capa_embedding):
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa Conv1D con Dropout
    cnn.add(Conv1D(32, 3, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    cnn.add(MaxPooling1D(2,1))
    # Segunda capa Conv1D con Dropout
    cnn.add(Conv1D(64, 5, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    cnn.add(MaxPooling1D(2,1))
    # Tercera capa Conv1D con Dropout
    cnn.add(Conv1D(128, 5, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    cnn.add(MaxPooling1D(2,1))
    # Cuarta capa Conv1D con Dropout
    cnn.add(Conv1D(256, 7, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    cnn.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    # Capa de salida
    cnn.add(Dense(3, activation='softmax'))

    return cnn

def cnn_bndp_four_f(capa_embedding):
    modelo = Sequential()
    modelo.add(capa_embedding)
    # Primera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(32, 3, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.5))
    modelo.add(MaxPooling1D(2,1))
    # Segunda capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(64, 5, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(MaxPooling1D(2,1))
    # Tercera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(128, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.5))
    modelo.add(MaxPooling1D(2,1))
    # Cuarta capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(256, 7, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.5))
    # Capa de salida
    modelo.add(Dense(3, activation='softmax'))

    return modelo

def cnn_bndp_four_fi(capa_embedding):

    modelo = Sequential()
    modelo.add(capa_embedding)

    # Primera capa Conv1D con Batch Normalization, Dropout y regularización L2
    modelo.add(Conv1D(32, 3, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.5))
    modelo.add(MaxPooling1D(2,1))

    # Segunda capa Conv1D con Batch Normalization, Dropout y regularización L2
    modelo.add(Conv1D(64, 5, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(MaxPooling1D(2,1))

    # Tercera capa Conv1D con Batch Normalization, Dropout y regularización L2
    modelo.add(Conv1D(128, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.5))
    modelo.add(MaxPooling1D(2,1))

    modelo.add(Conv1D(256, 7, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(GlobalMaxPooling1D())

    # Capa densa con Dropout y regularización L2
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.5))

    # Capa de salida
    modelo.add(Dense(3, activation='softmax'))

    return modelo


def cnn_bndp_four_ss(capa_embedding):
    modelo = Sequential()
    modelo.add(capa_embedding)
    # Primera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(32, 3, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.6))
    modelo.add(MaxPooling1D(2,1))
    # Segunda capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(64, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.6))
    modelo.add(MaxPooling1D(2,1))
    # Tercera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(128, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.6))
    modelo.add(MaxPooling1D(2,1))
    # Cuarta capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(256, 7, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.6))
    modelo.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.6))
    # Capa de salida
    modelo.add(Dense(3, activation='softmax'))

    return modelo
