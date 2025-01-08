#!/bin/bash

# EXPERIMENT 01
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 1. This experiment compares the performance of many different 
# vanilla machine learning methods (e.g. SVM, RF, etc.) on a small selection of datasets 
# for either HDC vectors or fingerprints.

FOLDER_PATH="/media/ssd/Programming/graph_hdc"

EXPERIMENTS_PATH="${FOLDER_PATH}/graph_hdc/experiments/fingerprints"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_01_a"
num_test="0.1"
size="2048"
depth="3"
datasets=("aqsoldb" "qm9_smiles" "clogp" "bbbp" "bace" "ames")
seeds=("1" "2" "3")
#seeds=("1")

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
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__hdc__${d}.py \\
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

# ~ Graph Neural Networks
# For graph neural networks we want to start experiments for all the different datasets and for
# each dataset we want to do multiple independent runs with different random seeds.

for d in "${datasets[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_01_hdc \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__gnn__${d}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'ex_01_hdc_${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
    "
done
done
