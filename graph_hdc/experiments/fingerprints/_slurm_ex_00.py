"""
This script will schedule all of the SLURM jobs for the hyperparameter optimization 
which constitutes "Experiment 0" to find the best hyperparameters for every combination
of encoding method, model type and dataset.
"""
import os
import pathlib
from itertools import product
from rich.pretty import pprint
import subprocess
from tqdm import tqdm
from auto_slurm.aslurmx import ASlurmSubmitter

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be use for the creation of the SLURM jobs.
AUTOSLURM_CONFIG = 'euler_2'

# The seed to be used for the random number generator and more importantly for the 
# dataset splitting! Will be the same for all expeirments.
SEED: int = 420

# The prefix for the experiment archives so that they can be easily identified later 
# on again.
PREFIX: str = 'ex_00_c'

# This is a data structure which maps each possible encoding method to a dictionary of the
# relevant hyperparameters and the corresponding values which are lists of values to
# be used for the hyperparameter sweep.
ENCODING_PARAMETER_MAP: dict[str, dict[str, list]] = {
    ('gnn', 'gnn'): {},
    ('morgan', 'fp'): {
        'FINGERPRINT_TYPE': ['morgan'],
        'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192],
        'FINGERPRINT_RADIUS': [1, 2, 3],
    },
    ('rdkit', 'fp'): {
        'FINGERPRINT_TYPE': ['rdkit'],
        'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192],
        'FINGERPRINT_RADIUS': [2, 3, 4], # This determines the maxPath parameter in this case
    },
    # ('atom', 'fp'): {
    #     'FINGERPRINT_TYPE': ['atom'],
    #     'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192],
    # },
    ('hdc', 'hdc'): {
        'EMBEDDING_SIZE': [1024, 2048, 4096, 8192],
        'NUM_LAYERS': [1, 2, 3],
    },
}

# This is a data structure which maps each possible model name to a dictionary of the 
# relevant hyperparameters and the corresponding values which are lists of values to 
# be used for the hyperparameter sweep.
MODEL_PARAMETER_MAP: dict[str, dict[str, list]] = {
    'random_forest': {
        'RF_NUM_ESTIMATORS': [10, 100, 500],
        'RF_MAX_DEPTH': [None, 10, 20],
        'RF_MAX_FEATURES': [None, 'sqrt'],
    },
    'grad_boost': {
        'GB_NUM_ESTIMATORS': [10, 100, 300],
        'GB_MAX_DEPTH': [2, 3, 4],
        #'GB_LEARNING_RATE': [0.01, 0.1, 0.2],
    },
    'k_neighbors': {
        'KN_NUM_NEIGHBORS': [3, 5, 10],
        'KN_WEIGHTS': ['uniform', 'distance'],
    },
    # 'linear': {
    #     'LN_ALPHA': [0.0001, 0.001, 0.01],
    #     'LN_FIT_INTERCEPT': [True, False],
    #     'LN_L1_RATIO': [0.0, 0.1, 0.5],
    # },
    'neural_net': {
        'NN_HIDDEN_LAYER_SIZES': [
            (10, 10),
            (50, 50),
            (100, 100),
        ],
        #'NN_ALPHA': [0.0001, 0.001, 0.01],
        'NN_LEARNING_RATE_INIT': [0.0001, 0.001, 0.01],
    },
    'neural_net2': {
        'NN_HIDDEN_LAYER_SIZES': [
            (10, 10),
            (50, 50),
            (100, 100),
        ],
        #'NN_ALPHA': [0.0001, 0.001, 0.01],
        'NN_LEARNING_RATE_INIT': [0.0001, 0.001, 0.01],
    },
    # 'gcn': {
    #     'CONV_UNITS': [
    #         (64, 64, 64),
    #         (128, 128, 128),
    #         (256, 256, 256),  
    #     ],
    #     'LEARNING_RATE': [0.001, 0.0001],
    #     'EPOCHS': [150],
    # },
    # 'gin': {
    #     'CONV_UNITS': [
    #         (64, 64, 64),
    #         (128, 128, 128),
    #         (256, 256, 256),  
    #     ],
    #     'LEARNING_RATE': [0.001, 0.0001],
    #     'EPOCHS': [150],
    # },
    # 'gatv2': {
    #     'CONV_UNITS': [
    #         (64, 64, 64),
    #         (128, 128, 128),
    #         (256, 256, 256),  
    #     ],
    #     'LEARNING_RATE': [0.001, 0.0001],
    #     'EPOCHS': [150],
    # },
}

# This is a data structure which maps each encoding method to a list of model names
# that are compatible with that encoding method. This is used to filter the model
# names when generating the SLURM jobs, so that only the models that are compatible
# with the encoding method are used in the hyperparameter sweep.
ENCODING_MODEL_MAP = {
    'gnn': ['gcn', 'gatv2', 'gin'],
    'fp': ['random_forest', 'grad_boost', 'k_neighbors', 'linear', 'neural_net', 'neural_net2'],
    'hdc': ['random_forest', 'grad_boost', 'k_neighbors', 'linear', 'neural_net', 'neural_net2'],
}

# This is a list of tuples where each tuple represents one dataset for which the experiment
# will be run. The first value in the tuple is the actual name of the dataset as it is represented 
# in the experiment files, the second value is a dictionary of parameters that are applied to 
# the experiment for that specific dataset.
DATASET_PARAMETER_TUPLES: list[tuple[str, dict]] = [
    # (
    #     'clogp', 
    #     {
    #         'NOTE': 'clogp',
    #     },
    # ),
    # (
    #     'aqsoldb', 
    #     {
    #         'NOTE': 'aqsoldb',
    #     },
    # ),
    # (
    #     'qm9_smiles', 
    #     {
    #         'NUM_DATA': 0.1,
    #         'NOTE': 'qm9_gap',
    #         'TARGET_INDEX': 7,  # GAP
    #     },
    # ),
    (
        'qm9_smiles', 
        {
            'NUM_DATA': 0.1,
            'NOTE': 'qm9_energy',
            'TARGET_INDEX': 10,  # U0
        },
    ),
]


if __name__ == "__main__":
    
    submitter = ASlurmSubmitter(config_name=AUTOSLURM_CONFIG)
    
    for (dataset, dataset_params) in DATASET_PARAMETER_TUPLES:
        
        for (name, encoding), encoding_params in ENCODING_PARAMETER_MAP.items():
            
            for model, model_params in MODEL_PARAMETER_MAP.items():
                
                if model not in ENCODING_MODEL_MAP[encoding]:
                    continue
                
                experiment_module = f'predict_molecules__{encoding}__{dataset}.py'
                experiment_path = os.path.join(PATH, experiment_module)
                
                # assert os.path.exists(experiment_module), (
                #     f'Experiment path "{experiment_path}" does not exist.'
                # )
                
                param_map: dict[str, list] = {
                    **encoding_params,
                    **model_params,
                }
                param_tuples = [
                    [(key, value) for value in values] 
                    for key, values in param_map.items()
                ]
                param_combinations = list(product(*param_tuples))
                print(f' ({encoding}, {model}, {dataset}) - {len(param_combinations)} combinations')

                for params in tqdm(param_combinations, desc=f"{encoding}/{model}/{dataset}"):
                    param_dict = dict(params)
                    param_dict.update(dataset_params)
                    
                    python_command_list = [
                        'python',
                        experiment_path,
                        f'--__DEBUG__=False ',
                        f'--__PREFIX__="{repr(PREFIX)}" ',
                        f'--SEED="{SEED}" ',
                        f'--MODELS="{repr([model])}" ',
                        f'--NUM_TEST="{repr(0.1)}" ',
                    ]
                    
                    for key, value in param_dict.items():
                        python_command_list.append(f'--{key}="{repr(value)}"')
                
                    python_command_string = ' '.join(python_command_list)

                    submitter.add_command(python_command_string)
                    # aslurm_command = [
                    #     'aslurmx',
                    #     '-cn', AUTOSLURM_CONFIG,
                    #     #'--dry-run',
                    #     'cmd',
                    #     python_command_string
                    # ]
                    # result = subprocess.run(aslurm_command, cwd=PATH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # if result.returncode != 0:
                    #     print(f"Warning: aslurmx command failed for {encoding}/{model}/{dataset} with params {param_dict}")
                    
    print('Submitting jobs to SLURM...')
    submitter.submit()