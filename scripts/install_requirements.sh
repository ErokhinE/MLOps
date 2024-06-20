#!/bin/bash

VENV_DIR="venv"

# Activate the virtual environment based on the OS
if [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    source $VENV_DIR/Scripts/activate
else
    # Unix-like systems
    source $VENV_DIR/bin/activate
fi

# Install the requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo "All requirements installed successfully."
else
    echo "requirements.txt not found."
fi