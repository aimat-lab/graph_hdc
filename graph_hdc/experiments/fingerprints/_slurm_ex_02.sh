#!/bin/bash

# EXPERIMENT 02
# This bash script implements the SLURM scheduling of all the individual experiment runs 
# required for Experiment 2. This experiment compares only a subset of the machine learning 
# methods but across a wider range of datasets. So in this case the datasets would be the 
# rows of the table and the methods the columns.


FOLDER_PATH="/media/ssd/Programming/graph_hdc"

EXPERIMENTS_PATH="${FOLDER_PATH}/graph_hdc/experiments/fingerprints"
PYTHON_PATH="${FOLDER_PATH}/venv/bin/python"

identifier="ex_02_a"
num_test="0.1"
size="2048"
depth="3"
datasets=("aqsoldb" "qm9_smiles" "clogp" "bbbp" "bace" "ames")
#seeds=("1" "2" "3" "4" "5")
seeds=("1")