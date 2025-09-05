"""
This script will schedule all of the SLURM jobs for "Experiment 1" 
to run the 
"""
import os
import sys
import json
import pathlib
from itertools import product
from rich.pretty import pprint
import subprocess
from tqdm import tqdm
from auto_slurm.aslurmx import ASlurmSubmitter

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be use for the creation of the SLURM jobs.
AUTOSLURM_CONFIG = 'euler_2'

# These are the different seeds to be used as the independent repetitions for the 
# the various computational experiments.
#SEEDS: list[int] = [0, 1, 2, 3, 4]
SEEDS: list[int] = [5, 6, 7, 8, 9]


# The prefix for the experiment archives so that they can be easily identified later 
# on again.
PREFIX: str = 'ex_01_aa'

# This is the path to the JSON file which contains all the hyperparameters that were 
# determined in the hyperparameter optimization.
HYPERPARAMETER_PATH: str = '/media/ssd2/Programming/graph_hdc/graph_hdc/experiments/fingerprints/experiment_best_parameters_map.json'

ENCODING_PARAMETER_TUPLES: list[tuple[str, str]] = [
    ('fp', {
        'FINGERPRINT_TYPE': 'morgan',
    }),
    ('fp', {
        'FINGERPRINT_TYPE': 'rdkit',
    }),
    ('gnn', {}),
    ('hdc', {}),
]

# This is a data structure which maps each encoding method to a list of model names
# that are compatible with that encoding method. This is used to filter the model
# names when generating the SLURM jobs, so that only the models that are compatible
# with the encoding method are used in the hyperparameter sweep.
ENCODING_MODEL_MAP = {
    'gnn': ['gatv2',],
    'fp': ['random_forest', 'grad_boost', 'k_neighbors', 'neural_net', 'neural_net2'],
    'hdc': ['random_forest', 'grad_boost', 'k_neighbors', 'neural_net', 'neural_net2'],
}

# If an entry for an encoding exists in this map, then those hyperparameters will be used instead 
# of the optimized ones from the hyperparameter optimization. This is mainly for the GNN and 
# HDC methods.
ENCODING_PARAMETER_MAP = {
    'gnn': {
        'CONV_UNITS': (128, 128, 128),
        'LEARNING_RATE': 0.001,
        'EPOCHS': 250,
    },
    'hdc': {
        'EMBEDDING_SIZE': 8192,
        'NUM_LAYERS': 2,
    },
}

# This is a list of tuples where each tuple represents one dataset for which the experiment
# will be run. The first value in the tuple is the actual name of the dataset as it is represented 
# in the experiment files, the second value is a dictionary of parameters that are applied to 
# the experiment for that specific dataset.
DATASET_PARAMETER_TUPLES: list[tuple[str, dict]] = [
    (
        'qm9_smiles', 
        {
            'NUM_DATA': 0.1,
            'NOTE': 'qm9_gap',
            'TARGET_INDEX': 7,  # GAP
        },
    ),
    (
        'qm9_smiles', 
        {
            'NUM_DATA': 0.1,
            'NOTE': 'qm9_energy',
            'TARGET_INDEX': 10,  # U0
        },
    ),
    (
        'clogp', 
        {
            'NOTE': 'clogp',
        },
    ),
    (
        'aqsoldb', 
        {
            'NOTE': 'aqsoldb',
        },
    ),
]


if __name__ == "__main__":
    
    # --- Loading optimized hyperparameters ---
    if os.path.exists(HYPERPARAMETER_PATH):
        with open(HYPERPARAMETER_PATH, 'r') as file:
            content: list = json.load(file)
            # key: (encoding, dataset, model)
            HYPERPARAMETER_MAP: dict[tuple, dict] = {
                tuple(element[0]): element[1]
                for element in content
            }
            
            pprint(HYPERPARAMETER_MAP)
        
    else:
        raise FileNotFoundError(f"Hyperparameter file not found: {HYPERPARAMETER_PATH}")
    
    # --- Counting experiments ---
    num_experiments = (
        len(ENCODING_PARAMETER_TUPLES) *
        max(len(l) for l in ENCODING_MODEL_MAP.values()) * 
        len(DATASET_PARAMETER_TUPLES) * 
        len(SEEDS)
    )
    print(f'Preparing to schedule {num_experiments} experiments for the ablation study.')

    # --- creating SLURM submitter ---
    submitter = ASlurmSubmitter(
        config_name=AUTOSLURM_CONFIG,
        batch_size=1,
        dry_run=False,
        randomize=True,
    )

    # --- submitting experiments ---
    with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:
    
        # Starting all of the actual slurm configs
        for dataset, dataset_params in DATASET_PARAMETER_TUPLES:
            
            for encoding, encoding_params in ENCODING_PARAMETER_TUPLES:
                
                models = ENCODING_MODEL_MAP[encoding]
                
                print(f'scheduling ({encoding}, {dataset}) - {len(models) * len(SEEDS)} experiments')
                
                for model in models:
                    
                    for seed in SEEDS:
                    
                        experiment_module = f'predict_molecules__{encoding}__{dataset}.py'
                        experiment_path = os.path.join(PATH, experiment_module)
                        
                        # If the encoding is found in the encoding parameter map, then use those parameters,
                        # otherwise use the hyperparameters from the hyperparameter optimization.
                        if encoding in ENCODING_PARAMETER_MAP:
                            optimal_params = ENCODING_PARAMETER_MAP[encoding]
                            
                        else:
                            _encoding = encoding
                            if 'FINGERPRINT_TYPE' in encoding_params:
                                _encoding = encoding_params['FINGERPRINT_TYPE']
                            
                            _dataset = dataset
                            if 'NOTE' in dataset_params:
                                _dataset = dataset_params['NOTE']
                                
                            optimal_params: dict[str, any] = HYPERPARAMETER_MAP[(_encoding, _dataset, model)]
                        
                        # constructing the standard command
                        python_command_list = [
                            'python',
                            experiment_path,
                            f'--__DEBUG__=False ',
                            f'--__PREFIX__="{repr(PREFIX)}" ',
                            f'--SEED="{seed}" ',
                            f'--MODELS="{repr([model])}" ',
                            f'--NUM_TEST="{repr(0.1)}" ',
                        ]
                        
                        # Adding the encoding specific parameters
                        for key, value in encoding_params.items():
                            python_command_list.append(f'--{key}="{repr(value)}"')
                        
                        # Adding the method specific optimized hyperparameters
                        for key, value in optimal_params.items():
                            python_command_list.append(f'--{key}="{repr(value)}"')
                    
                        # Adding the dataset specific parameters
                        for key, value in dataset_params.items():
                            python_command_list.append(f'--{key}="{repr(value)}"')

                        # Assembling the final string version of the command and submitting through
                        # Aslurm
                        python_command_string = ' '.join(python_command_list)
                    
                        submitter.add_command(python_command_string)
                        # aslurm_command = [
                        #     'aslurmx',
                        #     '-cn', AUTOSLURM_CONFIG,
                        #     #'--dry-run',
                        #     'cmd',
                        #     python_command_string
                        # ]
                        # #print(aslurm_command)
                        # result = subprocess.run(aslurm_command, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        # #print(result.stdout.decode('utf-8'), result.stderr.decode('utf-8'))
                        # if result.returncode != 0:
                        #     print(f"Warning: aslurmx command failed for {encoding}/{model}/{dataset} with params {optimal_params}")
                        
                        pbar.update(1)
                        
    print(f'Submitting {submitter.count_jobs()} jobs to SLURM...')
    submitter.submit()
