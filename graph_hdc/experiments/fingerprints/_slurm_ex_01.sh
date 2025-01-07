#!/bin/bash

# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for EXPERIMENT 01 which compares the performance of the HDC encoding and the 
# fingerprint encoding for a small selection of datasets.

PATH = "/media/ssd/Programming/graph_hdc"

# ~ Fingerprints

sbatch \
    --job-name=ex_01_fp \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="source ${PATH}/venv/bin/activate & python -m ${PATH}/experiments/fingerprints/predict_molecules__fp.py --DEBUG=\"False\""