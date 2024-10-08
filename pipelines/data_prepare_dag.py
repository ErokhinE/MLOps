from pendulum import datetime
from datetime import timedelta
import os
from airflow import DAG
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
PROJECT_PATH = os.environ['PROJECT_DIR']

with DAG(dag_id="data_preparation",
         start_date=datetime(2024, 7, 4, tz="UTC"),
         schedule_interval="*/5 * * * *",
         max_active_runs=1,
         catchup=False) as dag:
    
    wait_for_extract_sensor = ExternalTaskSensor(
        task_id='await_data_extraction',
        external_dag_id='extract_data',
        execution_delta=timedelta(hours=0),
        timeout=700,                       # Timeout in seconds
        allowed_states=['success'],        # Allowed states of the external task
        failed_states=['failed'],
        mode='poke',
        dag=dag,
    )
    prepare_data_command = f"python {PROJECT_PATH}/services/airflow/dags/data_prepare.py"
    prepare_data_task = BashOperator(
        task_id='prepare_data_task',
        bash_command=f"{prepare_data_command}"
    )

    wait_for_extract_sensor >> prepare_data_task
