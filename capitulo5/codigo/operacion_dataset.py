# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:37:12 2024

@author: PC
"""
import pandas as pd

def mezclar_dataset(ruta_dataset,valor_frac,valor_random):
    df = pd.read_csv(ruta_dataset)
    df = df.sample(frac=valor_frac, random_state=valor_random)  # Esto barajará aleatoriamente
    pd.to_csv(ruta_dataset)
    
def conteo_clases_dataset(dataset):
    clases_conteo = dataset['label'].value_counts()  # Cuenta las ocurrencias de cada clase
    clases_nombres = dataset['label'].unique()  # Obtiene los nombres únicos de las clases
    # Imprime la cantidad de clases y sus nombres
    print("Número de clases:", len(clases_conteo))
    print("Clases:", clases_conteo)
    print("Nombres de clases únicas:", clases_nombres)
    
def dividir_por_etiqueta(dataset,nom_columna, etiqueta):
    clase = dataset[dataset[nom_columna] == etiqueta]
    return clase

def dividir_por_columna(df,nom_columna):
    dividido = df[[nom_columna]]
    return dividido
    
def dividir_por_numero(dataset,num_ejemplares):
    df_mezcla = dataset.iloc[:num_ejemplares]
    return df_mezcla

def concatenar_vert(arreglo_datasets):
    df_final = pd.concat(arreglo_datasets, axis=0)
    return df_final

def concatenar_hori(arreglo_datasets):
    df_final = pd.concat(arreglo_datasets, axis=1)
    return df_final

def renombrar_columns(dataset,arreglo_nombres):
    dataset.columns= arreglo_nombres
    return dataset

def general_dataset(dataset, num_elementos):
    print("Numero de filas",dataset.shape[0]) #num filas de un dataset
    print("Primeros elementos",dataset.head(num_elementos)) #primeros n elementos de un dataset
    print("Numero de columnas", dataset.shape[1])
    print("Nombres de las columnas", dataset.columns)

