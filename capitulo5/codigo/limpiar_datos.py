# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:27:31 2023

@author: PC
"""
import re
import pandas as pd
from gestionarchivos import manejo_archivos

# primera fase de limpieza de documentos de texto extraidos de facebook
def limpiar_datos_primero(ruta_original, ruta_escrito):
    with open(ruta_original, 'r', encoding='utf-8') as original, open(ruta_escrito, 'w', encoding='utf-8') as escrito:
        for linea in original:
            linea = linea.lower()
            numeros = re.match(r'^\s*(\d{1,5})\s*$', linea)
            respuesta = re.match(r'^\s*(\d{1,5}) (respuesta|respuestas)\s*$', linea)
            gustar = re.match(r'^\s*me gusta\s*$', linea)
            fan = re.match(r'fan destacado\s*$', linea)
            editado = re.match(r'^\s*editado\s*$', linea)
            # puntos quita fechas, puntos, tiempo
            puntos = re.match(r'^\s*\.\s*$', linea)
            hora = re.match(r'^\d+:\d+ \/ \d+:\d+$', linea)
            vacio = re.match(r'^\s*$', linea)
            if not (numeros or respuesta or fan or editado or puntos or hora or gustar or vacio):
                escrito.write(linea)
# fase intermedia de limpieza de documentos de texto extraidos de facebook               
def limpiar_datos_intermedio(ruta_original, ruta_escrito):
    with open(ruta_original, 'r', encoding='utf-8') as original, open(ruta_escrito, 'w', encoding='utf-8') as escrito:
        for linea in original:
            gif = re.search(r'^\s*gif\s*$', linea)
            traduccion = re.match(r'^\s*ver traducción\s*$', linea)
            descripcion = re.match(r'^\s*no hay ninguna descripción de la foto disponible\s*$', linea)
            puntoyhora = re.match(r'\s*\·\s*', linea)
            signos = re.match(r'^\s*[.,;]\s*$', linea)
            comp = re.match(r'^\s*compartir\s*$', linea)
            autor = re.match(r'^\s*autor\s*$', linea)
            if not (gif or traduccion or descripcion or puntoyhora or signos or comp or autor):
                escrito.write(linea)
# segunda fase de limpieza de docuemntos de texto extraidos de facebook            
def limpiar_datos_segundo(ruta_original, ruta_escrito):
    bandera = False
    with open(ruta_original, 'r', encoding='utf-8', errors='ignore') as original:
        with open(ruta_escrito, 'w', encoding='utf-8', errors='ignore') as escrito:
            for linea in original:
                tiempo = re.match(r'^\s*(\d{1,3}) (sem|año|años|d|h)\s*$', linea)
                responder = re.match(r'^\s*responder\s*$', linea)
                if tiempo:
                  linea = ""
                  bandera = True
                else:
                  if bandera:
                    linea = ""
                    bandera = False
                  else:
                    if responder:
                      linea = ""
                escrito.write(linea)
                
def limpiar_datos_whatsApp(ruta_original, ruta_escrito):
    with open(ruta_original, 'r', encoding='utf-8') as original, open(ruta_escrito, 'w', encoding='utf-8') as escrito:
        for linea in original:
            linea = re.sub(r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{1,2}',"",linea)
            linea = re.sub(r'\s*(a|p)\. m\. - User \d{1,2}: ',"",linea)
            linea = re.sub('<multimedia omitido>',"", linea)
            linea = re.sub('se eliminó este mensaje',"", linea)
            linea = re.sub(r'@\d{9,15}\s*', "", linea)
            linea = linea.lower()
            if linea.strip():
                escrito.write(linea)
def offendes(ruta_csv,ruta_csv_nueva):
    df = pd.read_csv(ruta_csv)
    df['label'] = df['label'].replace({0: 1, 2: 0, 3: 0})
    df.head()
    manejo_archivos.crear_dataset_csv(df, ruta_csv_nueva)  
     
def limpiar_csv(ruta_csv,ruta_csv_nueva,funcion):
    df= pd.read_csv(ruta_csv)
    df['comentario'] = df['comentario'].apply(funcion)
    df = df.drop_duplicates()
    df = df.dropna()
    manejo_archivos.crear_dataset_csv(df, ruta_csv_nueva)
    
def filas_nulos(ruta_dataset):
    df= pd.read_csv(ruta_dataset)
    # Identificar las filas con valores nulos
    filas_con_nulos = df[df['comentario'].isnull()]
    # Mostrar las filas con valores nulos
    print("Filas con valores nulos:")
    print(filas_con_nulos)
    
def limpiarLabelStudio(ruta_csv, valor):
    df= pd.read_csv(ruta_csv)
    df = df.drop(columns=['annotation_id','annotator','created_at','lead_time','updated_at'])
    df['id']= df['id'] - valor
    df['sentiment'] = df['sentiment'].replace({'Ofensivo': 1, 'No ofensivo': 0, 'Grosero noofen': 2})
    df['label'] = df['label'].round().astype(int)
    manejo_archivos.crear_dataset_csv(ruta_csv)

