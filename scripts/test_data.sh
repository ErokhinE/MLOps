#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define paths
SAMPLE_SCRIPT="src.data"
SAMPLE_FUNCTION="sample_data"
SAMPLE_FILE="data/samples/sample.csv"
TESTS_DIR="tests"

# Step 1: Take a data sample using sample_data() from data.py
echo "Taking a data sample..."
python -c "
from $SAMPLE_SCRIPT import $SAMPLE_FUNCTION
$SAMPLE_FUNCTION()
"

# Step 2: Validate the data sample by running pytest
echo "Validating the data sample..."
pytest $TESTS_DIR
TEST_STATUS=$?

# Step 3: Version the data sample using DVC if tests pass
if [ $TEST_STATUS -eq 0 ]; then
    echo "Tests passed. Versioning the data sample..."

    # Add the versioned sample file to DVC and commit
    dvc add $SAMPLE_FILE
    git add $SAMPLE_FILE.dvc
    git commit -m "Versioned data sample: $SAMPLE_FILE"
    dvc push
else
    echo "Tests failed. Data sample will not be versioned."
    exit 1
fi

echo "Script completed successfully."
