#!/bin/bash

# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for EXPERIMENT 01 which compares the performance of the HDC encoding and the 
# fingerprint encoding for a small selection of datasets.

FOLDER_PATH="/media/ssd/Programming/graph_hdc"

EXPERIMENTS_PATH="${FOLDER_PATH}/graph_hdc/experiments/fingerprints"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="z"
num_test="0.9"
size="2048"
depth="3"
datasets=("aqsoldb" "qm9_smiles" "clogp" "bbbp" "bace" "ames")
#seeds=("1" "2" "3" "4" "5")
seeds=("1")

# ~ Fingerprints
# For fingerprints we want to start experiments for all the different datasets and for 
# each dataset we want to do multiple independent runs with different random seeds.

for d in "${datasets[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_01_fp \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__fp__${d}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'ex_01_fp_${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --FINGERPRINT_SIZE=\"${size}\" \\
        --FINGERPRINT_RADIUS=\"${depth}\" \\
    "
done
done

# ~ HDC Vectors
# For HDC vectors we want to start experiments for all the different datasets and for
# each dataset we want to do multiple independent runs with different random seeds.

for d in "${datasets[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_01_hdc \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__hdc.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'ex_01_hdc_${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --EMBEDDING_SIZE=\"${size}\" \\
        --NUM_LAYERS=\"${depth}\" \\
    "
done
done
