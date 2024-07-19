#!/bin/bash
DATA_VERSION=$( grep -A3 'data_version:' /mnt/c/Users/danil/Desktop/try_2/MLOps/configs/config.yaml | tail -n1 | awk '{ print $2}' )
SAMPLE_FILE="data/samples/sample.csv"
# Add the versioned sample file to DVC and commit
dvc add $SAMPLE_FILE
git add .
git commit -m "Versioned data sample: $SAMPLE_FILE"
git push
git tag -a "v$DATA_VERSION" -m "add data version v$DATA_VERSION"
git push --tags
dvc push
PYTHONCONFIGPATH="/mnt/c/Users/danil/Desktop/try_2/MLOps/scripts"
python $PYTHONCONFIGPATH/change_config.py