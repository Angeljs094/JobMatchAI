from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import pendulum
from datetime import timedelta
import sys
import pandas as pd
from machine_learning import train_and_save_model


# Agregar el directorio '/home/angel/airflow/etl' al sys.path
etl_path = '/home/angel/airflow/etl'
if etl_path not in sys.path:
    sys.path.append(etl_path)

import etl_functions

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'computrabajo_etl_streamlit',
    default_args=default_args,
    description='DAG for ETL and launching Streamlit app',
    schedule=timedelta(days=1),
    start_date=pendulum.today('UTC').add(days=-1),
    catchup=False,
) as dag:

    def extract(**kwargs):
        full_path = etl_functions.extract_csv()
        if full_path:
            kwargs['ti'].xcom_push(key='full_path', value=full_path)

    def transform(**kwargs):
        full_path = kwargs['ti'].xcom_pull(key='full_path')
        if full_path:
            df = etl_functions.transform_csv(full_path)
            if df is not None:
                kwargs['ti'].xcom_push(key='dataframe', value=df.to_dict())

    def load(**kwargs):
        full_path = kwargs['ti'].xcom_pull(key='full_path')
        df_dict = kwargs['ti'].xcom_pull(key='dataframe')
        if df_dict:
            df = pd.DataFrame.from_dict(df_dict)
            etl_functions.load_csv(df, full_path)

    def train_model():
        data_path = '/home/angel/airflow/computrabajo_filter/computrabajo_jobs.csv'
        output_dir = '/home/angel/airflow/etl'
        train_and_save_model(data_path, output_dir)

    extract_task = PythonOperator(
        task_id='extract',
        python_callable=extract,
        provide_context=True
    )

    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform,
        provide_context=True
    )

    load_task = PythonOperator(
        task_id='load',
        python_callable=load,
        provide_context=True
    )

    train_task = PythonOperator(
        task_id='train_task',
        python_callable=train_model,
        provide_context=True
    )


    launch_streamlit_app = BashOperator(
        task_id='launch_streamlit_app',
        bash_command='streamlit run /home/angel/airflow/etl/app.py',
        retries=3,
    )

    extract_task >> transform_task >> load_task >> train_task >> launch_streamlit_app
