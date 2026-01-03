#!/bin/bash
# Script to run training in the correct conda environment
conda run --no-capture-output -n piu_yolo python training.py "$@"
