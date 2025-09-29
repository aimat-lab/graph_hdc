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
from auto_slurm.aslurmx import ASlurmSubmitter


PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be use for the creation of the SLURM jobs.
AUTOSLURM_CONFIG = 'haicore_1gpu'

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
    ('hdc', {'MODELS': ['neural_net2'], 'EMBEDDING_SIZE': 8192, 'NUM_LAYERS': 2}),
    ('fp', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'morgan', 'FINGERPRINT_SIZE': 8192, 'FINGERPRINT_RADIUS': 2}),
    ('fp', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'rdkit', 'FINGERPRINT_SIZE': 8192, 'FINGERPRINT_RADIUS': 2}),
    ('fp', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'atom', 'FINGERPRINT_SIZE': 8192, 'FINGERPRINT_RADIUS': 2}),
    ('fp', {'MODELS': ['neural_net2'], 'FINGERPRINT_TYPE': 'torsion', 'FINGERPRINT_SIZE': 8192, 'FINGERPRINT_RADIUS': 2}),
]

DATASET_PARAM_TUPLES: dict[str, dict] = [
    ('aqsoldb', {'NOTE': 'aqsoldb_logs'}),
    ('clogp', {'NOTE': 'clogp'}),
    ('freesolv', {'TARGET_INDEX': 0, 'DATASET_NAME': 'freesolv', 'DATASET_TYPE': 'regression', 'NOTE': 'freesolv_hfe'}),
    ('lipophilicity', {'TARGET_INDEX': 0, 'DATASET_NAME': 'lipophilicity', 'DATASET_TYPE': 'regression', 'NOTE': 'lipophilicity_logD'}),
    ('bace_reg', {'TARGET_INDEX': 0, 'DATASET_NAME': 'bace_reg', 'DATASET_TYPE': 'regression', 'NOTE': 'bace_ic50'}),
    ('hopv15_exp', {'TARGET_INDEX': 2, 'DATASET_NAME': 'hopv15_exp', 'DATASET_TYPE': 'regression', 'NOTE': 'hopv15_gap'}),
    ('hopv15_exp', {'TARGET_INDEX': 3, 'DATASET_NAME': 'hopv15_exp', 'DATASET_TYPE': 'regression', 'NOTE': 'hopv15_jsc'}),
    ('hopv15_exp', {'TARGET_INDEX': 4, 'DATASET_NAME': 'hopv15_exp', 'DATASET_TYPE': 'regression', 'NOTE': 'hopv15_voc'}),
    ('hopv15_exp', {'TARGET_INDEX': 5, 'DATASET_NAME': 'hopv15_exp', 'DATASET_TYPE': 'regression', 'NOTE': 'hopv15_pce'}),
    ('compas', {'TARGET_INDEX': 3, 'NOTE': 'compas_dipole'}),
    ('compas', {'TARGET_INDEX': 2, 'NOTE': 'compas_gap'}),
    ('compas', {'TARGET_INDEX': 4, 'NOTE': 'compas_energy'}),
    ('qm9_smiles', {'TARGET_INDEX': 3, 'NOTE': 'qm9_dipole'}),
    ('qm9_smiles', {'TARGET_INDEX': 4, 'NOTE': 'qm9_alpha'}),
    ('qm9_smiles', {'TARGET_INDEX': 7, 'NOTE': 'qm9_gap'}),
    ('qm9_smiles', {'TARGET_INDEX': 10, 'NOTE': 'qm9_energy'}),
    ('qm9_smiles', {'TARGET_INDEX': 9, 'NOTE': 'qm9_zpve'}),
    ('qm9_smiles', {'TARGET_INDEX': 12, 'NOTE': 'qm9_enthalpy'}),
    ('qm9_smiles', {'TARGET_INDEX': 14, 'NOTE': 'qm9_cv'}),
    # ('tadf', {'TARGET_INDEX': 0, 'NOTE': 'tadf_rate'}), 
    # ('tadf', {'TARGET_INDEX': 1, 'NOTE': 'tadf_splitting'}),
    # ('tadf', {'TARGET_INDEX': 2, 'NOTE': 'tadf_oscillator'}),
    # ('zinc250', {'TARGET_INDEX': 0, 'NOTE': 'zinc_clogp'}), 
    # ('zinc250', {'TARGET_INDEX': 1, 'NOTE': 'zinc_qed'}),
    # ('zinc250', {'TARGET_INDEX': 2, 'NOTE': 'zinc_sas'}),
]

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


num_experiments = (
    len(ENCODING_PARAM_TUPLES) * 
    len(DATASET_PARAM_TUPLES) * 
    len(SEEDS)
)
print(f'Preparing to schedule {num_experiments} experiments for the ablation study.')

# --- creating SLURM submitter ---
submitter = ASlurmSubmitter(
    config_name=AUTOSLURM_CONFIG,
    batch_size=10,
    randomize=True,
    overwrite_fillers={
        'time': '12:00:00', # 12 hours should be enough
        'venv': '/home/iti/tm4030/Programming/graph_hdc/.venv',
        'partition': 'normal',
    },
)

# --- submitting experiments ---
with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:
    
    for (encoding, encoding_param_dict) in ENCODING_PARAM_TUPLES:
        
        for (dataset_name, dataset_param_dict) in DATASET_PARAM_TUPLES:
            
            for seed in SEEDS:
            
                ## --- Assemble Python Command ---
                # At first we need to assemble the python command that will be executed within each 
                # SLURM job. For this we need to call the correct experiment module with the correct 
                # parameters.
                experiment_module = f'predict_molecules__{encoding}__{dataset_name}.py'
                experiment_path = os.path.join(PATH, experiment_module)
                if not os.path.exists(experiment_path):
                    experiment_module = f'predict_molecules__{encoding}.py'
                    experiment_path = os.path.join(PATH, experiment_module)
                
                python_command_list = [
                    'python',
                    experiment_path,
                    f'--__DEBUG__=False ',
                    f'--__PREFIX__="{repr(PREFIX)}" ',
                    f'--SEED="{seed}" ',
                    f'--NUM_TEST="{repr(0.1)}" ',
                    f'--NUM_TRAIN="{repr(1.0)}" ',
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
