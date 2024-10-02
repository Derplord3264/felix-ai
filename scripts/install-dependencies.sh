#!/bin/bash

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install transformers flask torch

echo "Dependencies installed successfully."
