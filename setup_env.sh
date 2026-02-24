#!/bin/bash

# Create virtual environment
echo "Creating virtual environment 'penv'..."
python3 -m venv penv

# Activate environment and install requirements
echo "Installing dependencies..."
source penv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "---"
echo "Setup complete! To start using the environment, run:"
echo "source penv/bin/activate"
