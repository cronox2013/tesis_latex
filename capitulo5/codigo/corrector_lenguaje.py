# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:42:46 2023

@author: PC
"""
import re
import pandas as pd
from gestionarchivos import manejo_archivos


def reemplazar_abreviaciones(texto_con_errores):
    abreviaciones = {
        r'\b(qe|q|k|ke)\b': "que",
        r'\b(x|for)\b': "por",
        r'\b(xq|porq|pq|pque|poq)\b': "porque",
        r'\bd\b': "de",
        r'\b(pndj|pdjo|pnjo|pjo)\b': "pendejo",
        r'\b(pndjs|pdjos|pnjos|pjos)\b': "pendejos",
        r'\b(pta|p4ta)\b': "puta",
        r'\b(crjo|carjo|cjo|crajo|crj)\b': "carajo",
        r'\b(hjdp|hdp|hdpt)\b': "hijo de puta",
        r'\b(cjdo|cjd)\b': "cojudo",
        r'\bwtf\b': "what the fuck",
        r'\bhdpm\b': "hijo de tu puta madre",
        r'\balv\b': "a la verga"
    }

    for patron, reemplazo in abreviaciones.items():
        texto_con_errores = re.sub(patron, reemplazo, texto_con_errores, flags=re.IGNORECASE)

    return texto_con_errores
       
def eliminar_urls(texto):
    # Patrón para reconocer URLs
    link = re.sub(r'https?://\S+|www\.\S+|noticias\.\S+',"",texto)
    return link

def eliminar_car_individual(texto):
    texto = re.sub(r'\s+[a-z]\s+'," ", texto)
    return texto
def eliminar_car_especial(texto):
    texto = re.sub(r'[^\w\s]','', texto)
    return texto
def eliminar_car_indi_ini(texto):
    texto = re.sub(r'^[a-z]\s+',"",texto)
    return texto

def eliminar_numero(texto):
    texto= re.sub(r'\d+',"",texto)
    return texto
def eliminar_espacios(texto):
    texto = re.sub(r'\s+',' ', texto)
    return texto

def separar_colum_offendes(ruta_csv,ruta_csv_nueva):
    df = pd.read_csv(ruta_csv)
    
    columnas_deseadas = ['comment', 'label']
    df_copy = df[columnas_deseadas].copy()
    df_copy.rename(columns={'comment': 'comentario'}, inplace=True)
    manejo_archivos.crear_dataset_csv(df_copy, ruta_csv_nueva)

def limpiar_csv_total(ruta_csv,ruta_csv_nueva,ruta_txt):
    df= pd.read_csv(ruta_csv)
    df = df.drop_duplicates()
    df['comentario']= df['comentario'].astype(str)
    df['comentario']= df['comentario'].str.lower()
    df['comentario'] = df['comentario'].apply(reemplazar_abreviaciones)
    df['comentario'] = df['comentario'].apply(eliminar_urls)
    df['comentario'] = df['comentario'].apply(eliminar_car_especial)
    df['comentario'] = df['comentario'].apply(eliminar_car_indi_ini)
    df['comentario'] = df['comentario'].apply(eliminar_numero)
    df['comentario'] = df['comentario'].apply(eliminar_espacios)
    df['comentario'] = df['comentario'].apply(eliminar_car_individual)
    df = df.drop_duplicates()
    df = df.dropna()
    manejo_archivos.escribir_archivo(ruta_txt, ruta_csv_nueva+" contiene:"+ str(len(df))+" comentarios en total\n")
    manejo_archivos.crear_dataset_csv(df, ruta_csv_nueva)