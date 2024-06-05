# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:57:07 2024

@author: PC
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.layers import MaxPooling1D

def cnn_two(capa_embedding):
  cnn = Sequential()
  cnn.add(capa_embedding)
  cnn.add(Conv1D(64, 3, activation='relu'))
  cnn.add(MaxPooling1D(2,1))
  cnn.add(Conv1D(128, 5, activation='relu'))
  cnn.add(GlobalMaxPooling1D())
  cnn.add(Dense(128, activation='relu'))
  cnn.add(Dense(3, activation='softmax'))

  return cnn

def cnn_dp_two(capa_embedding):
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa Conv1D con Dropout
    cnn.add(Conv1D(32, 3, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(MaxPooling1D(2,2))
    # Segunda capa Conv1D con Dropout
    cnn.add(Conv1D(64, 5, activation='relu'))
    cnn.add(Dropout(0.3))  # Añade Dropout con tasa del 30%
    cnn.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))  # Añade Dropout con tasa del 50%
    # Capa de salida
    cnn.add(Dense(3, activation='softmax'))

    return cnn

def cnn_dp_two1(capa_embedding):
    cnn = Sequential()
    cnn.add(capa_embedding)
    # Primera capa Conv1D con Dropout
    cnn.add(Conv1D(32, 3, activation='relu'))
    cnn.add(Dropout(0.1))  # Añade Dropout con tasa del 30%
    cnn.add(MaxPooling1D(2,1))
    # Segunda capa Conv1D con Dropout
    cnn.add(Conv1D(64, 5, activation='relu'))
    cnn.add(Dropout(0.1))  # Añade Dropout con tasa del 30%
    cnn.add(GlobalMaxPooling1D())
    # Capa densa con Dropout
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dropout(0.2))  # Añade Dropout con tasa del 50%
    # Capa de salida
    cnn.add(Dense(3, activation='softmax'))

    return cnn
