"""
This script will schedule all of the SLURM jobs for Experiment 04,  
which will perform an ablation study over different dataset sizes to find 
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
SEED: int = 0

SEEDS: list[int] = [0, 1, 2,]

# The prefix for the experiment archives so that they can be easily identified later 
# on again.
PREFIX: str = 'ex_04_e'

# This determines the dataset that is used for the ablation study.
DATASET_NAME: str = 'qm9_smiles'

# This list determines the combinations of encoding methods and specific prediction 
# models to be used. The whole ablation study will be performed for each of these
# combinations individually.
ENCODING_METHOD_TUPLES: list[tuple[str, str]] = [
    ('gnn', 'gatv2'),
    #('gnn', 'gin'),
    ('hdc', 'neural_net'),
    ('hdc', 'random_forest'),
    #('fp', 'neural_net'),
    #('fp', 'random_forest'),
    # ('hdc', 'k_neighbors'),
    # ('fp', 'k_neighbors'),
]

# This maps from the encoding method to a dictionary of parameters that are specific to 
# that encoding method.
ENCODING_PARAMETERS_MAP: dict[str, dict] = {
    'gnn': {
        'NUM_DATA': 1.0, # use the whole dataset
        'EPOCHS': 150,
        'TARGET_INDEX': 10,
    },
    'hdc': {
        'NUM_DATA': 1.0, # use the whole dataset
        'RF_NUM_ESTIMATORS': 100,
        'RF_MAX_DEPTH': 10,
        'RF_MAX_FEATURES': 'sqrt',   
        'NN_HIDDEN_LAYER_SIZES': (100, 100),
        'NN_ALPHA': 0.001,
        'NN_LEARNING_RATE_INIT': 0.001,
        'EMBEDDING_SIZE': 2048,
        'NUM_LAYERS': 2,
        'TARGET_INDEX': 10,
    },
    'fp': {
        'NUM_DATA': 1.0, # use the whole dataset
        'RF_NUM_ESTIMATORS': 100,
        'RF_MAX_DEPTH': 10,
        'RF_MAX_FEATURES': 'sqrt',   
        'NN_HIDDEN_LAYER_SIZES': (100, 100),
        'NN_ALPHA': 0.001,
        'NN_LEARNING_RATE_INIT': 0.001,
        'FINGERPRINT_SIZE': 2048,
        'FINGERPRINT_RADIUS': 2, 
        'TARGET_INDEX': 10,   
    },
}

# This data structure determines the dataset sizes to be used for the ablation study.
NUM_TRAIN_SWEEP: list[int | float] = [
    10,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    7000,
    10_000,
    50_000,
    100_000,
]

num_experiments = len(ENCODING_METHOD_TUPLES) * len(NUM_TRAIN_SWEEP) * len(SEEDS)
print(f'Preparing to schedule {num_experiments} experiments for the ablation study.')

with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:
    
    for (encoding, model) in ENCODING_METHOD_TUPLES:
        
        for num_train in NUM_TRAIN_SWEEP:
            
            for seed in SEEDS:
                
                ## --- Assemble Python Command ---
                # At first we need to assemble the python command that will be executed within each 
                # SLURM job. For this we need to call the correct experiment module with the correct 
                # parameters.
                experiment_module = f'predict_molecules__{encoding}__{DATASET_NAME}.py'
                experiment_path = os.path.join(PATH, experiment_module)
                
                python_command_list = [
                    'python',
                    experiment_path,
                    f'--__DEBUG__=False ',
                    f'--__PREFIX__="{repr(PREFIX)}" ',
                    f'--SEED="{seed}" ',
                    f'--MODELS="{repr([model])}" ',
                    f'--NUM_TEST="{repr(0.1)}" ',
                    f'--NUM_TRAIN="{repr(num_train)}" ',
                ]
                
                # The parameters to be used for this specific encoding method.
                param_dict = ENCODING_PARAMETERS_MAP[encoding]
                for key, value in param_dict.items():
                    python_command_list.append(f'--{key}="{repr(value)}"')
                    
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
