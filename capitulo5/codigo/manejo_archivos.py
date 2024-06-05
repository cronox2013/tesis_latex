# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:31:45 2023

@author: PC
"""
import csv
import pandas as pd
import os

def leer_archivo(ruta):
  with open(ruta, 'r') as original:
      for linea in original:
        print(linea)
        
def escribir_archivo(rutita, texto):
    # Abrir un archivo en modo de aprendizaje ('a')
    with open(rutita, 'a') as archivo:
    # Escribir el nuevo contenido al final del archivo
        archivo.write(texto)

      
def escribir_archivo_vacio(rutita, texto):
    with open(rutita, 'w') as escrito:
      escrito.write(texto)

def vaciar_archivo(ruta):
  with open(ruta, 'w') as archivo:
    pass  # No se escribe ningún contenido
  print("El archivo se ha vaciado.")
  
def copiar_archivo(original, vacio):
  with open(original, 'r') as original:
      with open(vacio, 'w') as escrito:
        for linea in original:
            escrito.write(linea)
      original.close()
      escrito.close()
      
def vaciar_csv(ruta_csv):
  with open(ruta_csv, 'w', newline='') as archivo:
    escritor_csv = csv.writer(archivo)
    
def crear_csv_vacio(ruta):
  data = {}
  dataset = pd.DataFrame(data)
  dataset.to_csv(ruta, index=False)

def crear_dataset_vacio_colum(nombre_columna):
    df_vacio = pd.DataFrame(columns=[nombre_columna])
    return df_vacio

def crear_dataset_csv(data, ruta):
  dataset = pd.DataFrame(data)
  dataset.to_csv(ruta, index=False)
  
def crear_dataset_excel(data, ruta):
  dataset = pd.DataFrame(data)
  dataset.to_excel(ruta, index=False)
  
def crear_csv_dataset(ruta_csv):
    dataframe = pd.read_csv(ruta_csv)
    return dataframe

def crear_csv_excel(ruta_csv,ruta_excel):
    df = pd.read_csv(ruta_csv)
# Guardar el DataFrame como un archivo de Excel
    df.to_excel(ruta_excel, index=False)
    
def crear_txt_vacio(ruta):
    with open(ruta, 'w') as archivo:
        pass
    
def crear_carpeta_vacia(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

def recorrer_archivo(ruta_carpeta):
    archivos = os.listdir(ruta_carpeta)
    for archivo in archivos:
     print(archivo)
    