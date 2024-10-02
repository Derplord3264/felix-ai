#!/bin/bash

# Install dependencies
pip install datasets transformers flask

# Download dataset
python3 data/download_dataset.py

# Train model
python3 model/train_model.py

# Run web app
python3 web/app.py
