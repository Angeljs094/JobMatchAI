import os
import sys

# Asegúrate de que la carpeta machine_learning esté en el sys.path
sys.path.append(os.path.dirname(__file__))

# Importar la función train_and_save_model desde el archivo job_classificacion_con_grid
from job_classificacion_con_grid import train_and_save_model

# Ruta de los datos y el directorio de salida
data_path = '/home/angel/airflow/computrabajo_filter/computrabajo_jobs.csv'
output_dir = os.path.dirname(__file__)  # Guardar en la carpeta actual

# Ejecutar la función
train_and_save_model(data_path, output_dir)
