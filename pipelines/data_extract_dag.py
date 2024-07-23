from airflow import DAG, AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from pendulum import datetime as pdt
import os
from data import sample_data, validate_initial_data

PROJECT_PATH = os.environ['PROJECT_DIR']


# Define DAG parameters
dag_args = {
    'dag_id': 'extract_data',
    'start_date': pdt(2024, 7, 4, tz="UTC"),
    'schedule_interval': '*/5 * * * *',
    'catchup': False,
}

# Instantiate the DAG
with DAG(**dag_args) as dag:

    # Task to sample data
    def task_sample_data():
        original_path = os.getcwd()
        os.chdir(PROJECT_PATH)
        sampled_data = sample_data()
        os.chdir(original_path)
        if sampled_data is None:
            raise AirflowException("Data sampling failed")
        return "Sampled Successfully"

    sample_data_op = PythonOperator(
        task_id='task_sample_data',
        python_callable=task_sample_data
    )

    # Task to validate data
    def task_validate_data():
        validation_success = validate_initial_data()
        if not validation_success:
            raise AirflowException("Data validation failed")
        return "Validated Successfully"

    validate_data_op = PythonOperator(
        task_id='task_validate_data',
        python_callable=task_validate_data
    )

    # Define the bash command for versioning data
    os.chdir(PROJECT_PATH)
    versioning_script = f"{PROJECT_PATH}/scripts/versioning_sample.sh"
    if not os.path.exists(versioning_script.strip()):
        raise Exception(f"Script {versioning_script} not found")

    version_data_op = BashOperator(
        task_id='task_version_data',
        bash_command='bash {{ params.script_path }}',
        params={'script_path': versioning_script},
    )

    # Set task dependencies
    sample_data_op >> validate_data_op >> version_data_op



    
