"""
Environment check of run_ex27.sbatch (Experiment 27): imports the experiment modules of the runs (without running them)
and the packages of the Optuna stage, so that a library missing on a compute node stops the job before its first run.
On JUPITER, the Booster compute nodes lack system libraries that the login nodes have (libcairo for cairosvg, the X11
libraries for RDKit's drawing extension), and both are imported at startup by chem_mat_data.

    python check_imports_ex27.py      (from the fingerprints folder)
"""
import os
import sys
import importlib.util

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH)

for name in ('predict_molecules', 'predict_molecules__hdc', 'predict_molecules__fp'):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PATH, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    # pycomex looks the experiment modules up in sys.modules
    sys.modules[name] = module
    spec.loader.exec_module(module)

import optuna  # noqa: E402,F401
import torch  # noqa: E402
import sklearn.ensemble  # noqa: E402,F401  (gradient boosting and random forest of ex_28)
import _optuna_ex_28  # noqa: E402,F401  (the study driver of ex_28 and its scheduler module)
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402,F401
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402,F401

print(f'imports ok (torch {torch.__version__}, CUDA available: {torch.cuda.is_available()})')
