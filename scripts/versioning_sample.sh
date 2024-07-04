#!/bin/bash
DATA_VERSION=$( grep -A3 'data_version:' configs/config.yaml | tail -n1 | awk '{ print $2}' )
# Add the versioned sample file to DVC and commit
dvc add $SAMPLE_FILE
git add .
git commit -m "Versioned data sample: $SAMPLE_FILE"
git push
git tag -a "v$DATA_VERSION" -m "add data version v$DATA_VERSION"
git push --tags
dvc push
PYTHONCONFIGPATH="$PWD/scripts"
python $PYTHONCONFIGPATH/change_config.py