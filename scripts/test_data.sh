#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define paths
SAMPLE_FUNCTION="sample_data"
SAMPLE_FILE="data/samples/sample.csv"
TESTS_DIR="tests"

export PYTHONPATH="$PWD/src"

# Loop from 1 to 5
for i in {1..5}
do
    echo "Iteration $i:"

    # Step 1: Take a data sample using sample_data() from data.py
    echo "Taking a data sample..."
    python $PYTHONPATH/data.py

    # Step 2: Validate the data sample by running pytest
    echo "Validating the data sample..."
    pytest $TESTS_DIR
    TEST_STATUS=$?

    # Step 3: Version the data sample using DVC if tests pass
    if [ $TEST_STATUS -eq 0 ]; then
        DATA_VERSION=$( grep -A3 'data_version:' configs/config.yaml | tail -n1 | awk '{ print $2}' )
        echo "Tests passed. Versioning the data sample..."
        # Add the versioned sample file to DVC and commit
        dvc add $SAMPLE_FILE
        git add .
        git commit -m "Versioned data sample: $SAMPLE_FILE (Iteration $i)"
        git push
        git tag -a "v$DATA_VERSION-iteration-$i" -m "add data version v$DATA_VERSION for iteration $i"
        git push --tags
        dvc push
        PYTHONCONFIGPATH="$PWD/scripts"
        python $PYTHONCONFIGPATH/change_config.py
    else
        echo "Tests failed in iteration $i. Data sample will not be versioned."
        exit 1
    fi

    echo "Iteration $i completed successfully."
done

echo "Script completed successfully."
