#!/bin/bash

# Define the virtual environment directory
VENV_DIR="venv"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "All requirements installed successfully."

# Deactivate the virtual environment
deactivate
