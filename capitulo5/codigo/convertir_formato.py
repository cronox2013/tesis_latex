# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:03:01 2023

@author: PC
"""

import csv
import os
import pandas as pd

from limpiezadataset import limpiar_datos
from gestionarchivos import manejo_archivos

def convertir_txt_csv(ruta_txt, ruta_csv):
    with open(ruta_txt, 'r', encoding='utf-8', errors='ignore') as archivo_texto:
        lineas = archivo_texto.readlines()
    
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        # Escribir el encabezado
        escritor_csv.writerow(["comentario"])
        for linea in lineas:
            campos = [linea.strip()]
            # Escribe los campos en el archivo CSV
            escritor_csv.writerow(campos)
        print(f"Se ha convertido el archivo '{ruta_txt}' a '{ruta_csv}' en formato CSV.")

# realiza el uso de la funcion de limpieza de whatsApp sobre
# todos los archivos que se encuentren en la carpeta correspondiente    
def todo_convertir_whats(carpeta_original, carpeta_nueva):
    archivos = os.listdir(carpeta_original)
    numero = 1
    s="/"
    for archivo in archivos:
        nombre_nueva = carpeta_nueva+str(numero)
        manejo_archivos.crear_carpeta_vacia(nombre_nueva)
        manejo_archivos.crear_txt_vacio(nombre_nueva+"/ruta1.txt")
        limpiar_datos.limpiar_datos_whatsApp(carpeta_original+s+archivo, nombre_nueva+"/ruta1.txt")
        convertir_txt_csv(nombre_nueva+"/ruta1.txt", nombre_nueva+"/ruta2.csv")
        numero+=1

# realiza el uso de la funcion de limpieza de facebook sobre
# todos los archivos que se encuentren en la carpeta correspondiente 
def todo_convertir(carpeta_original,carpeta_nueva):
    archivos = os.listdir(carpeta_original)
    numero = 1
    s="/"
    for archivo in archivos:
        nombre_nueva = carpeta_nueva+str(numero)
        manejo_archivos.crear_carpeta_vacia(nombre_nueva)
        manejo_archivos.crear_txt_vacio(nombre_nueva+"/ruta1.txt")
        limpiar_datos.limpiar_datos_primero(carpeta_original+s+archivo, nombre_nueva+"/ruta1.txt" )
        manejo_archivos.crear_txt_vacio(nombre_nueva+"/ruta2.txt")
        limpiar_datos.limpiar_datos_intermedio(nombre_nueva+"/ruta1.txt", nombre_nueva+"/ruta2.txt")
        manejo_archivos.crear_txt_vacio(nombre_nueva+"/ruta3.txt")
        limpiar_datos.limpiar_datos_segundo(nombre_nueva+"/ruta2.txt", nombre_nueva+"/ruta3.txt")
        manejo_archivos.crear_csv_vacio(nombre_nueva+"/ruta4.csv")
        convertir_txt_csv(nombre_nueva+"/ruta3.txt", nombre_nueva+"/ruta4.csv")
        numero+=1
        
def unir_dataset(carpeta_original, nombre_csv):
    archivos = os.listdir(carpeta_original)
    s="/"
    big_dataset = manejo_archivos.crear_dataset_vacio_colum('comentario')
    ruta_txt = carpeta_original+"/info.txt"
    informacion =[]
    for archivo in archivos:
        data = manejo_archivos.crear_csv_dataset(carpeta_original+s+archivo+s+"ruta4.csv")
        big_dataset = pd.concat([big_dataset, data], ignore_index=True)
        informacion.append(archivo+" tamanio original: "+str(len(data)))
    manejo_archivos.crear_dataset_csv(big_dataset, carpeta_original+s+nombre_csv)
    informacion.append(nombre_csv+" contiene un total de: "+str(len(big_dataset))+" comentarios en total")
    manejo_archivos.crear_txt_vacio(ruta_txt)
    for info in informacion:
        manejo_archivos.escribir_archivo(ruta_txt, info+"\n")
        
def unir_datos(ruta_csv,ruta_csv1,ruta_csv2,ruta_excel):
    df = manejo_archivos.crear_csv_dataset(ruta_csv)
    df1= manejo_archivos.crear_csv_dataset(ruta_csv1)
    df2= manejo_archivos.crear_csv_dataset(ruta_csv2)
    df3 = pd.concat([df,df1,df2],ignore_index=True)
    manejo_archivos.crear_dataset_excel(df3, ruta_excel)
    
def reemplazar_valor2(ruta_answer,ruta_old,ruta_nueva):
    df = pd.read_csv(ruta_answer)
    df1 = pd.read_csv(ruta_old)
    for idx in range(len(df)):
    # Obtenemos el valor de la columna 'x' en el DataFrame df en la fila 'idx'
        valor = df.at[idx, 'id']
        df1.iloc[valor]['label']= df.at[idx,'sentiment']
    manejo_archivos.crear_dataset_csv(df1,ruta_nueva)
    
def enumerar_dataset(ruta_csv,ruta_nueva):
    df = pd.read_csv(ruta_csv)
    df['id'] = df.reset_index().index
    manejo_archivos.crear_dataset_csv(df,ruta_nueva)

def reemplazar_valor(ruta_answer, ruta_old, ruta_nueva):
    # Leer los DataFrames desde los archivos CSV
    df = pd.read_csv(ruta_answer)
    df1 = pd.read_csv(ruta_old)
    # Iterar sobre las filas del DataFrame df
    for idx, row in df.iterrows():
        # Obtener el valor de 'sentiment' en la fila actual de df
        sentiment_value = row['sentiment']
        # Obtener el valor de 'id' en la fila actual de df
        id_value = row['id']
        # Reemplazar el valor de 'label' en el DataFrame df1
        df1.loc[df1['id'] == id_value, 'label'] = sentiment_value
    # Guardar el DataFrame modificado en un nuevo archivo CSV
    df1.to_csv(ruta_nueva, index=False)  