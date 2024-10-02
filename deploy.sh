#!/bin/bash

# Install dependencies
pip install datasets transformers flask

# Download dataset
python data/download_dataset.py

# Train model
python model/train_model.py

# Run web app
python web/app.py
