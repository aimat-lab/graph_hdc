#!/bin/bash

# EXPERIMENT 03
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 3. This experiment performs an ablation study on the dimensionality 
# of the embedding. The goal is to get a plot where the x axis is different embedding sizes 
# and the y-axis is the performance of one of the machine learning methods. This will be 
# compared for the HDC vectors and the fingerprints.


FOLDER_PATH="/media/ssd/Programming/graph_hdc"

EXPERIMENTS_PATH="${FOLDER_PATH}/graph_hdc/experiments/fingerprints"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_03_a"
dataset="qm9_smiles"
depth="3"
num_test="0.1"
seeds=("1" "2" "3")
embedding_sizes=("64" "128" "256" "512" "1024" "2048" "4096" "8192")

# ~ Fingerprints
# For fingerprints we want to start experiments for all the different datasets and for 
# each dataset we want to do multiple independent runs with different random seeds.

for e in "${embedding_sizes[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_03_fp \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__fp__${dataset}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --FINGERPRINT_SIZE=\"${e}\" \\
        --FINGERPRINT_RADIUS=\"${depth}\" \\
        --NUM_TRAIN=\"${t}\" \\
    "
done
done

# ~ HDC Vectors
# For HDC vectors we want to start experiments for all the different datasets and for
# each dataset we want to do multiple independent runs with different random seeds.

for e in "${embedding_sizes[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_03_hdc \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__hdc__${dataset}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --EMBEDDING_SIZE=\"${e}\" \\
        --NUM_LAYERS=\"${depth}\" \\
    "
done
done