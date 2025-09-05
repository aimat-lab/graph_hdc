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
SEEDS: list[int] = [0, 1, 2, 3, 4]

# The prefix for the experiment archives so that they can be easily identified later 
# on again.
PREFIX: str = 'ex_02_a'

# This list determines the combinations of encoding methods and specific prediction 
# models to be used. The whole ablation study will be performed for each of these
# combinations individually.
ENCODING_METHOD_TUPLES: list[tuple[str, str]] = [
    ('gnn', 'gatv2'),
    ('hdc', 'neural_net2'),
    #('hdc', 'random_forest'),
    ('fp', 'neural_net2'),
    #('fp', 'random_forest'),
]
ENCODING_PARAM_TUPLES: list[tuple[str, dict]] = [
    ('gnn', {'MODELS': ['gatv2']}),
    ('hdc', {'MODELS': ['neural_net2']}),
    ('morgan', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'morgan'}),
    ('rdkit', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'rdkit'}),
    ('atom', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'atom'}),
    ('torsion', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'torsion'}),
]

DATASET_PARAM_TUPLES: dict[str, dict] = [
    ('aqsoldb', {'NOTE': 'aqsoldb_logs'}),
    ('lipop', {'NOTE': 'lipop_logd'}),
    ('zinc250', {'TARGET_INDEX': 0, 'NOTE': 'zinc_clogp'}), 
    ('zinc250', {'TARGET_INDEX': 1, 'NOTE': 'zinc_qed'}),
    ('zinc250', {'TARGET_INDEX': 2, 'NOTE': 'zinc_sas'}),
    ('qm9_smiles', {'TARGET_INDEX': 3, 'NOTE': 'qm9_dipole'}),
    ('qm9_smiles', {'TARGET_INDEX': 4, 'NOTE': 'qm9_alpha'}),
    ('qm9_smiles', {'TARGET_INDEX': 7, 'NOTE': 'qm9_gap'}),
    ('qm9_smiles', {'TARGET_INDEX': 10, 'NOTE': 'qm9_energy'}),
    ('compas', {'TARGET_INDEX': 3, 'NOTE': 'compas_dipole'}),
    ('compas', {'TARGET_INDEX': 2, 'NOTE': 'compas_gap'}),
    ('compas', {'TARGET_INDEX': 4, 'NOTE': 'compas_energy'}),
    ('tadf', {'TARGET_INDEX': 0, 'NOTE': 'tadf_rate'}), 
    ('tadf', {'TARGET_INDEX': 1, 'NOTE': 'tadf_splitting'}),
    ('tadf', {'TARGET_INDEX': 2, 'NOTE': 'tadf_oscillator'}),
]

ENCDODING_SIZE_PARAMETER_MAP: dict[str, str] = {
    'fp': 'FINGERPRINT_SIZE',
    'hdc': 'EMBEDDING_SIZE',   
}
ENCODING_DEPTH_PARAMETER_MAP: dict[str, str] = {
    'fp': 'FINGERPRINT_RADIUS',
    'hdc': 'NUM_LAYERS',
}

# This maps from the encoding method to a dictionary of parameters that are specific to 
# that encoding method.
ENCODING_PARAMETERS_MAP: dict[str, dict] = {
    'gnn': {
        'NUM_DATA': 1.0, # use the whole dataset
        'EPOCHS': 150
    },
    'hdc': {
        'NUM_DATA': 1.0, # use the whole dataset
        'RF_NUM_ESTIMATORS': 100,
        'RF_MAX_DEPTH': 10,
        'RF_MAX_FEATURES': 'sqrt',   
        'NN_HIDDEN_LAYER_SIZES': (100, 100),
        'NN_ALPHA': 0.001,
        'NN_LEARNING_RATE_INIT': 0.001,
    },
    'fp': {
        'NUM_DATA': 1.0, # use the whole dataset
        'RF_NUM_ESTIMATORS': 100,
        'RF_MAX_DEPTH': 10,
        'RF_MAX_FEATURES': 'sqrt',   
        'NN_HIDDEN_LAYER_SIZES': (100, 100),
        'NN_ALPHA': 0.001,
        'NN_LEARNING_RATE_INIT': 0.001,  
    },
}

EMBEDDING_SIZE = 8192
EMBEDDING_DEPTH = 2

num_experiments = (
    len(ENCODING_PARAM_TUPLES) * 
    len(DATASET_PARAM_TUPLES) * 
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
    
    for (encoding, encoding_param_dict) in ENCODING_METHOD_TUPLES:
        
        for (dataset_name, dataset_param_dict) in DATASET_PARAM_TUPLES:
            
            for embedding_depth in EMBEDDING_DEPTH_SWEEP:
            
                for seed in SEEDS:
                
                    ## --- Assemble Python Command ---
                    # At first we need to assemble the python command that will be executed within each 
                    # SLURM job. For this we need to call the correct experiment module with the correct 
                    # parameters.
                    experiment_module = f'predict_molecules__{encoding}__{dataset_name}.py'
                    experiment_path = os.path.join(PATH, experiment_module)
                    
                    embedding_size_parameter = ENCDODING_SIZE_PARAMETER_MAP[encoding]
                    embedding_depth_parameter = ENCODING_DEPTH_PARAMETER_MAP[encoding]
                    
                    python_command_list = [
                        'python',
                        experiment_path,
                        f'--__DEBUG__=False ',
                        f'--__PREFIX__="{repr(PREFIX)}" ',
                        f'--SEED="{seed}" ',
                        f'--NUM_TEST="{repr(0.1)}" ',
                        f'--NUM_TRAIN="{repr(1.0)}" ',
                        f'--{embedding_size_parameter}="{repr(EMBEDDING_SIZE)}" ',
                        f'--{embedding_depth_parameter}="{repr(EMBEDDING_DEPTH)}" '
                    ]
                    
                    # The parameters to be used for this specific encoding method.
                    param_dict = ENCODING_PARAMETERS_MAP[encoding]
                    
                    param_dict.update(encoding_param_dict)
                    param_dict.update(dataset_param_dict)
                    
                    for key, value in param_dict.items():
                        python_command_list.append(f'--{key}="{repr(value)}"')
                        
                    python_command_string = ' '.join(python_command_list)
                        
                    submitter.add_command(python_command_string)
                    pbar.update(1)
                        
    print(f'Submitting {submitter.count_jobs()} jobs to SLURM...')
    submitter.submit()
