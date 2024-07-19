#!/bin/bash

# Ensure the dags directory exists
if [ ! -d "./services/airflow/dags" ]; then
  mkdir -p ./services/airflow/dags
fi

# Iterate through each .py file in the pipelines directory
for py_file in ./pipelines/*.py; do
  # Get the base name of the file
  base_name=$(basename "$py_file")
  # Create the symbolic link in the dags directory
  ln -sf "./pipelines/$base_name" "./services/airflow/dags/$base_name"
done
