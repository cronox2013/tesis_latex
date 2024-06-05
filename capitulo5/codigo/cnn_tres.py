# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:41:45 2024

@author: PC
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, BatchNormalization, Dropout
from keras import regularizers
from tensorflow.keras.layers import MaxPooling1D


def cnn_base_tt(capa_embedding):
  cnn = Sequential()
  cnn.add(capa_embedding)
  cnn.add(Conv1D(32, 3, activation='relu'))
  cnn.add(MaxPooling1D(2,2))
  cnn.add(Conv1D(64, 5, activation='relu'))
  cnn.add(MaxPooling1D(2,2))
  cnn.add(Conv1D(128, 5, activation='relu'))
  cnn.add(GlobalMaxPooling1D())
  cnn.add(Dense(128, activation='relu'))
  cnn.add(Dense(3, activation='softmax'))

  return cnn

def cnn_base_bn_tt(capa_embedding):
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa convolucional con BatchNormalization
    cnn.add(Conv1D(32, 3, activation='relu'))
    cnn.add(BatchNormalization())  # Añadir batch normalization
    cnn.add(MaxPooling1D(2,2))
    # Segunda capa convolucional con BatchNormalization
    cnn.add(Conv1D(64, 5, activation='relu'))
    cnn.add(BatchNormalization())  # Añadir batch normalization
    cnn.add(MaxPooling1D(2,2))
    # Tercera capa convolucional con BatchNormalization
    cnn.add(Conv1D(128, 5, activation='relu'))
    cnn.add(BatchNormalization())  # Añadir batch normalization
    # Global max pooling
    cnn.add(GlobalMaxPooling1D())
    # Capa densa intermedia con BatchNormalization
    cnn.add(Dense(128, activation='relu'))
    cnn.add(BatchNormalization())  # Añadir batch normalization
    # Capa de salida con función de activación softmax
    cnn.add(Dense(3, activation='softmax'))

    return cnn

def cnn_base_dp_tt(capa_embedding):
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa Conv1D con Dropout
    cnn.add(Conv1D(32, 3, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(MaxPooling1D(2,2))
    # Segunda capa Conv1D con Dropout
    cnn.add(Conv1D(64, 5, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(MaxPooling1D(2,2))
    # Tercera capa Conv1D con Dropout
    cnn.add(Conv1D(128, 5, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    # Capa de salida
    cnn.add(Dense(3, activation='softmax'))

    return cnn

def cnn_base_l2_tt(capa_embedding):
    # Configura la regularización L2 con un factor adecuado (0.05 en este caso)
    l2_regularizer = regularizers.l2(0.05)
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa Conv1D con regularización L2
    cnn.add(Conv1D(32, 3, activation='relu', kernel_regularizer=l2_regularizer))
    cnn.add(MaxPooling1D(2,2))
    # Segunda capa Conv1D con regularización L2
    cnn.add(Conv1D(64, 5, activation='relu', kernel_regularizer=l2_regularizer))
    cnn.add(MaxPooling1D(2,2))
    # Tercera capa Conv1D con regularización L2
    cnn.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2_regularizer))
    # GlobalMaxPooling1D
    cnn.add(GlobalMaxPooling1D())
    # Capa densa intermedia con regularización L2
    cnn.add(Dense(128, activation='relu', kernel_regularizer=l2_regularizer))
    # Capa de salida
    cnn.add(Dense(3, activation='softmax'))

    return cnn

def cnn_base_bndp_128t(capa_embedding):
    modelo = Sequential()
    modelo.add(capa_embedding)
    # Primera capa Conv1D con Batch Normalization, Dropout 
    modelo.add(Conv1D(32, 3, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(MaxPooling1D(2,2))
    # Segunda capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(64, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(MaxPooling1D(2,2))
    # Tercera capa Conv1D con Batch Normalization, Dropout 
    modelo.add(Conv1D(128, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.5))

    # Capa de salida
    modelo.add(Dense(3, activation='softmax'))

    return modelo

def cnn_base_bndp_64t(capa_embedding):
    modelo = Sequential()
    modelo.add(capa_embedding)
    # Primera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(32, 3, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(MaxPooling1D(2,2))
    # Segunda capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(64, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(MaxPooling1D(2,2))
    # Tercera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(128, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dropout(0.5))
    # Capa de salida
    modelo.add(Dense(3, activation='softmax'))

    return modelo

def cnn_base_bndp_32t(capa_embedding):
    modelo = Sequential()
    modelo.add(capa_embedding)
    # Primera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(32, 3, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(MaxPooling1D(2,2))
    # Segunda capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(64, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(MaxPooling1D(2,2))
    # Tercera capa Conv1D con Batch Normalization, Dropout
    modelo.add(Conv1D(128, 5, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    modelo.add(GlobalMaxPooling1D())
    # Capa densa con Dropout y regularización L2
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dropout(0.5))
    # Capa de salida
    modelo.add(Dense(3, activation='softmax'))

    return modelo

