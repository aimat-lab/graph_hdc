"""
This script will schedule all of the SLURM jobs for Experiment 04,  
which will perform an ablation study over different embedding sizes to find 
out about the data efficiency of the different encoding methods compared 
specifically to the GNN methods which are expected to be worse at that.
"""
import os
import pathlib
from itertools import product
from rich.pretty import pprint
import subprocess
from tqdm import tqdm

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be use for the creation of the SLURM jobs.
AUTOSLURM_CONFIG = 'euler_2'

# The seed to be used for the random number generator and more importantly for the 
# dataset splitting! Will be the same for all expeirments.
SEEDS: list[int] = [0, 1, 2, 3, 4]

# The prefix for the experiment archives so that they can be easily identified later 
# on again.
PREFIX: str = 'ex_05_a'

# This determines the dataset that is used for the ablation study.
DATASET_NAME: str = 'aqsoldb'

NUM_TRAIN: float = 1.0

# This data structure determines the dataset sizes to be used for the ablation study.
EMBEDDING_SIZE_SWEEP: list[int | float] = [
    250,
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10_000,
    15_000,
    20_000,
    25_000,
    30_000,
    35_000,
    40_000,
    45_000,
    50_000,
]

num_experiments = len(EMBEDDING_SIZE_SWEEP) * len(SEEDS)
print(f'Preparing to schedule {num_experiments} experiments for the ablation study.')

with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:
        
    for emb_size in EMBEDDING_SIZE_SWEEP:
        
        for seed in SEEDS:
            
            ## --- Assemble Python Command ---
            # At first we need to assemble the python command that will be executed within each 
            # SLURM job. For this we need to call the correct experiment module with the correct 
            # parameters.
            experiment_module = f'reconstruct_molecules.py'
            experiment_path = os.path.join(PATH, experiment_module)
            
            python_command_list = [
                'python',
                experiment_path,
                f'--__DEBUG__=False ',
                f'--__PREFIX__="{repr(PREFIX)}" ',
                f'--SEED="{seed}" ',
                f'--DATASET_NAME="{DATASET_NAME}" ',
                f'--EMBEDDING_SIZE="{repr(emb_size)}" ',
                f'--NUM_DATA="{repr(NUM_TRAIN)}" ',
            ]
                
            python_command_string = ' '.join(python_command_list)
            
            ## --- Assemble ASLURM Command ---
            # Finally, on the top level we need to call the ASLURMX command to schedule the job.
            aslurm_command = [
                'aslurmx',
                '-cn', AUTOSLURM_CONFIG,
                #'--dry-run',
                'cmd',
                python_command_string
            ]
            result = subprocess.run(aslurm_command, cwd=PATH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                print(f"Warning: aslurmx command failed!")
                
            pbar.update(1)
