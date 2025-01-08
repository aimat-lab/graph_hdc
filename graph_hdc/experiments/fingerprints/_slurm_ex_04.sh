#!/bin/bash

# EXPERIMENT 04
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 4. This experiment performs an ablation study on size of the 
# dataset. The goal is to get a plot where the x axis is different dataset sizes and the 
# y-axis is the performance of one of the machine learning methods. This will be compared 
# for the HDC vectors and the graph neural networks. The expectation is that for very small 
# datasets the graph neural networks will perform poorly compared to the HDC vectors but 
# at some point the graph neural networks will outperform the HDC vectors.

FOLDER_PATH="/media/ssd/Programming/graph_hdc"

EXPERIMENTS_PATH="${FOLDER_PATH}/graph_hdc/experiments/fingerprints"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_04_a"
dataset="qm9_smiles"
size="2048"
depth="3"
num_test="0.1"
seeds=("1" "2" "3")
train_sizes=("0.01" "0.1" "0.2" "0.5" "0.75" "1")

# ~ HDC Vectors
# For HDC vectors we want to start experiments for a sweep of different dataset sizes and 
# for different seeds for each dataset size.

for t in "${train_sizes[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_04_hdc \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__hdc__${dataset}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --EMBEDDING_SIZE=\"${size}\" \\
        --NUM_LAYERS=\"${depth}\" \\
        --NUM_TRAIN=\"${t}\" \\
    "
done
done


# ~ Graph Neural Networks
# For graph neural networks we want to start experiments for a sweep of different dataset
# sizes and for different seeds for each dataset size.

for t in "${train_sizes[@]}"; do
for s in "${seeds[@]}"; do
sbatch \
    --job-name=ex_04_gnn \
    --mem=90GB \
    --time=01:00:00 \
    --wrap="${PYTHON_PATH} ${EXPERIMENTS_PATH}/predict_molecules__gnn__${dataset}.py \\
        --__DEBUG__=\"False\" \\
        --__PREFIX__=\"'${identifier}'\" \\
        --SEED=\"${s}\" \\
        --IDENTIFIER=\"'${identifier}'\" \\
        --NUM_TEST=\"${num_test}\" \\
        --NUM_TRAIN=\"${t}\" \\
    "
done
done
