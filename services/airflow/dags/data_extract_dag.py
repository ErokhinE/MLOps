from airflow import DAG, AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from pendulum import datetime as pdt
import os
from data import sample_data, validate_initial_data




# Define DAG parameters
dag_args = {
    'dag_id': 'extract_data',
    'start_date': pdt(2024, 7, 4, tz="UTC"),
    'schedule_interval': '*/30 * * * *',
    'catchup': False,
}

# Instantiate the DAG
with DAG(**dag_args) as dag:
    
    # Task to sample data
    def sample_task():
        path = os.getcwd()
        os.chdir("/mnt/c/Users/danil/Desktop/try_2/MLOps")
        data_sample = sample_data()
        os.chdir(path)
        if data_sample is None:
            raise AirflowException("Data sampling failed")
        return "Sampled Successfully"

    sample_op = PythonOperator(
        task_id='sample_data_task',
        python_callable=sample_task
    )

    # Task to validate data
    def validation_task():
        is_valid = validate_initial_data()
        if not is_valid:
            raise AirflowException("Data validation failed")
        return "Validated Successfully"

    validate_op = PythonOperator(
        task_id='validate_data_task',
        python_callable=validation_task
    )

    # Define the bash command for versioning data
    os.chdir("/mnt/c/Users/danil/Desktop/try_2/MLOps")
    version_command = "/mnt/c/Users/danil/Desktop/try_2/MLOps/scripts/versioning_sample.sh"
    if not os.path.exists(version_command.strip()):
        raise Exception(f"Script {version_command} not found")

    version_op = BashOperator(
        task_id='version_data_task',
        bash_command='bash {{ params.script_path }}',
        params={'script_path': '/mnt/c/Users/danil/Desktop/try_2/MLOps/scripts/versioning_sample.sh'},
    )

    # Set task dependencies
    sample_op >> validate_op >> version_op




    
