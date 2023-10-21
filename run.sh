#!/bin/bash

# Set the directory where your Jupyter Notebook is located
cd ToRun

# Activate your Python environment (if needed)
# source activate your_environment

# Convert the notebook to a script
jupyter nbconvert --to script run.ipynb

# Run the converted script
python run.py