import pandas as pd
import os
from datetime import timedelta

# Directorios
path = "/home/angel/airflow/computrabajo_job"
output_path = "/home/angel/airflow/computrabajo_filter"
output_file = os.path.join(output_path, "computrabajo_jobs.csv")

def extract_csv():

    

    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    
    # Verificar si algún archivo correspondiente a cualquier mes existe
    files = os.listdir(path)
    for month in months:
        csv_filename = f"computrabajo_{month}.csv"
        if csv_filename in files:
            full_path = os.path.join(path, csv_filename)
            print(f"Archivo encontrado: {full_path}")
            return full_path

    # Verificar si el archivo existe
    files = os.listdir(path)
    if csv_filename in files:
        full_path = os.path.join(path, csv_filename)
        print(f"Archivo encontrado: {full_path}")
        return full_path
    else:
        print(f"No se encontró el archivo {csv_filename}")
        return None

def transform_csv(full_path):
    try:
        print(f"Cargando el archivo CSV: {full_path}")
        df = pd.read_csv(full_path)
        print("Archivo cargado exitosamente")

        # Eliminar columnas no deseadas
        columns_to_drop = ["company_link", "date_scraped", "industry", "subtitle", "salary", "contract_type"]
        df.drop(columns=columns_to_drop, inplace=True)
        print("Columnas eliminadas")

        # Borrar la palabra clave
        df['keywords'] = df['keywords'].str.replace("Palabras clave:", "", regex=False)
        print("Palabras clave eliminadas")

        def is_valid_date(date_str):
            try:
                date = pd.to_datetime(date_str, format='%Y-%m-%d')
                return date.year != 2023
            except ValueError:
                return False

        # Filtrar el DataFrame para conservar solo las filas con fechas válidas y que no pertenecen a 2023
        df = df[df['date_posted'].apply(is_valid_date)]
        print("Fechas filtradas")
        return df

    except Exception as e:
        print(f"Error al procesar el archivo {full_path}: {e}")
        return None

def load_csv(df, full_path):
    # Crear el directorio de salida si no existe
    os.makedirs(output_path, exist_ok=True)
    print(f"Directorio de salida verificado: {output_path}")

    # Guardar el archivo filtrado
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
        print(f"Archivo procesado y guardado en {output_file}")
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Archivo procesado y agregado a {output_file}")

    # Eliminar el archivo original
    os.remove(full_path)
    print(f"Archivo original eliminado: {full_path}")
