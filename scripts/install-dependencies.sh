#!/bin/bash

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# Install Python dependencies
pip install transformers flask torch datasets

echo "Dependencies installed successfully."
