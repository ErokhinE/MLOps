#!/bin/bash

# Check all versions 0 до 4
cd $PROJECT_DIR
for i in {1..5}
do
    echo "Evaluation champion with data-version $i"
    mlflow run . --env-manager=local -e predict -P data_version=$i -P docker_port=5152
done