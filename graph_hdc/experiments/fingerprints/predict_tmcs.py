"""
Base experiment for property prediction on transition metal complexes (TMCs).

Extends :mod:`predict_molecules` by overriding the data-loading, filtering, and
default featurisation hooks to handle the decomposed TMC format provided by
``chem_mat_data``.  The model training and evaluation pipeline is inherited
unchanged.

The default ``process_dataset`` hook computes a simple baseline representation
(mean-pooled 91-dim ``node_attributes`` per complex).  Child experiments
(e.g. ``predict_tmcs__hdc.py``) override this hook with richer encodings.

Target properties
-----------------

The tmQMg dataset provides 20 regression targets.  The ``TARGET_INDEX``
parameter selects which one to predict (default 2 = ``tzvp_homo_lumo_gap``):

=====  =====================================
Index  Property
=====  =====================================
0      tzvp_lumo_energy
1      tzvp_homo_energy
2      tzvp_homo_lumo_gap
3      homo_lumo_gap_delta
4      tzvp_electronic_energy
5      electronic_energy_delta
6      tzvp_dispersion_energy
7      dispersion_energy_delta
8      enthalpy_energy
9      enthalpy_energy_correction
10     gibbs_energy
11     gibbs_energy_correction
12     zpe_correction
13     heat_capacity
14     entropy
15     tzvp_dipole_moment
16     dipole_moment_delta
17     polarisability
18     lowest_vibrational_frequency
19     highest_vibrational_frequency
=====  =====================================
"""
import os
import random
from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from pycomex.functional.experiment import Experiment

RDLogger.DisableLog('rdApp.*')
from pycomex.utils import folder_path, file_namespace
from chem_mat_data.tmc import load_tmc_dataset
from chem_mat_data.tmc_processing import MetalOrganicProcessing

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the TMC dataset. Used to download via ``load_tmc_dataset()``.
DATASET_NAME: str = 'tmqmg'
# :param DATASET_NAME_ID:
#       Identifier for results; usually same as ``DATASET_NAME``.
DATASET_NAME_ID: str = DATASET_NAME
# :param DATASET_TYPE:
#       All tmQMg targets are continuous.
DATASET_TYPE: str = 'regression'
# :param TARGET_INDEX:
#       Index into the 20-element target vector. 2 = tzvp_homo_lumo_gap.
TARGET_INDEX: Union[int, None] = 2
# :param NUM_TEST:
#       Fraction of the dataset held out for testing.
NUM_TEST: Union[int, float] = 0.1

# :param TARGET_COLUMNS:
#       The 20 tmQMg target column names in order.  This list determines
#       which DataFrame columns are collected as ``graph_labels`` and in
#       which order.
TARGET_COLUMNS: List[str] = [
    'tzvp_lumo_energy',
    'tzvp_homo_energy',
    'tzvp_homo_lumo_gap',
    'homo_lumo_gap_delta',
    'tzvp_electronic_energy',
    'electronic_energy_delta',
    'tzvp_dispersion_energy',
    'dispersion_energy_delta',
    'enthalpy_energy',
    'enthalpy_energy_correction',
    'gibbs_energy',
    'gibbs_energy_correction',
    'zpe_correction',
    'heat_capacity',
    'entropy',
    'tzvp_dipole_moment',
    'dipole_moment_delta',
    'polarisability',
    'lowest_vibrational_frequency',
    'highest_vibrational_frequency',
]

# Atomic numbers of common donor elements for connecting-atom inference.
_DONOR_ELEMENTS = {7, 8, 15, 16, 34, 9, 17, 35, 53}

# == EXPERIMENT SETUP ==

experiment = Experiment.extend(
    'predict_molecules.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# ---------------------------------------------------------------------------
# Helper: infer connecting atoms when not stored in the CSV
# ---------------------------------------------------------------------------

def _infer_connecting_atoms(ligand_smiles: str) -> List[int]:
    """
    Heuristically determine donor atom indices in a single ligand SMILES.

    Prioritises negatively charged atoms, then neutral heteroatoms from
    :data:`_DONOR_ELEMENTS`, falling back to index 0 for carbon-donor
    ligands like CO.

    :param ligand_smiles: SMILES string of a single ligand fragment.
    :returns: List of 0-based atom indices of inferred donor atoms.
    """
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(ligand_smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(
                    mol,
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                )
            except Exception:
                pass
    if mol is None:
        return [0]

    if mol.GetNumAtoms() == 1:
        return [0]

    charged = []
    neutral = []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() < 0:
            charged.append(atom.GetIdx())
        elif atom.GetAtomicNum() in _DONOR_ELEMENTS:
            neutral.append(atom.GetIdx())

    return charged or neutral or [0]


# ---------------------------------------------------------------------------
# Hook overrides
# ---------------------------------------------------------------------------

@experiment.hook('load_dataset', replace=True, default=False)
def load_dataset(e: Experiment):
    """
    Load a TMC dataset via ``load_tmc_dataset`` and process each row through
    ``MetalOrganicProcessing`` to produce graph dicts.

    The resulting ``index_data_map`` has the same structure as the organic
    molecule pipeline so that downstream hooks (splitting, evaluation, etc.)
    work unchanged.
    """
    e.log(f'loading TMC dataset "{e.DATASET_NAME}"...')
    df = load_tmc_dataset(e.DATASET_NAME, folder_path='/tmp')
    e.log(f'loaded DataFrame with {len(df)} rows.')

    processing = MetalOrganicProcessing()
    index_data_map = {}
    skipped = 0

    for row_idx, row in df.iterrows():

        # --- ligand SMILES ---
        lig_smi = row['ligand_smiles']
        if isinstance(lig_smi, str):
            # Dot-separated string — split into individual ligands
            lig_smi = lig_smi.split('.')

        # --- connecting atom indices ---
        if 'connecting_atom_indices' in row.index and isinstance(row['connecting_atom_indices'], list):
            conn_indices = row['connecting_atom_indices']
        else:
            conn_indices = [_infer_connecting_atoms(s) for s in lig_smi]

        # --- target values ---
        targets = []
        skip = False
        for col in e.TARGET_COLUMNS:
            if col in row.index:
                val = row[col]
                if np.isnan(val):
                    skip = True
                    break
                targets.append(float(val))
        if skip or len(targets) == 0:
            skipped += 1
            continue

        # --- process graph ---
        try:
            graph = processing.process(
                metal=row['metal'],
                ligand_smiles=lig_smi,
                connecting_atom_indices=conn_indices,
                oxidation_state=int(row.get('oxidation_state', 0)),
                total_charge=int(row.get('total_charge', 0)),
                spin_multiplicity=int(row.get('spin_multiplicity', 1)),
                graph_labels=np.array(targets, dtype=float),
            )
            index_data_map[len(index_data_map)] = graph
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                e.log(f'  skipped row {row_idx}: {exc}')
            continue

    e.log(f'processed {len(index_data_map)} TMCs ({skipped} skipped)')

    # --- Sub-sampling (same logic as parent) ---
    if e.NUM_DATA is not None:
        if isinstance(e.NUM_DATA, int):
            num_data = e.NUM_DATA
        elif isinstance(e.NUM_DATA, float):
            num_data = int(e.NUM_DATA * len(index_data_map))

        random.seed(e.SEED)
        index_data_map = dict(
            random.sample(list(index_data_map.items()), k=num_data)
        )

    metadata = {}
    return index_data_map, metadata


@experiment.hook('filter_dataset', replace=True, default=False)
def filter_dataset(e: Experiment, index_data_map: dict) -> None:
    """
    Filter TMC graphs: remove entries with too few nodes, no edges, or NaN targets.

    This replaces the organic-molecule filter that relies on SMILES validation.
    """
    e.log(f'filtering TMC dataset...')
    e.log(f'starting with {len(index_data_map)} samples...')

    indices = list(index_data_map.keys())
    for index in indices:
        graph = index_data_map[index]

        if len(graph['node_indices']) < 2:
            del index_data_map[index]
            continue

        if len(graph['edge_indices']) == 0:
            del index_data_map[index]
            continue

        labels = graph['graph_labels']
        if np.any(np.isnan(labels)):
            del index_data_map[index]
            continue

    e.log(f'finished filtering with {len(index_data_map)} samples remaining.')


@experiment.hook('process_dataset', replace=True, default=True)
def process_dataset(e: Experiment, index_data_map: dict) -> None:
    """
    Default TMC featurisation: mean-pooled ``node_attributes`` (91-dim).

    This provides a simple baseline.  The HDC child experiment overrides
    this with :class:`~graph_hdc.models.HyperNet` encoding.
    """
    e.log('computing mean-pooled node-attribute features (91-dim baseline)...')
    for index, graph in index_data_map.items():
        node_attrs = graph['node_attributes']  # (num_atoms, 91)
        graph['graph_features'] = np.mean(node_attrs, axis=0).astype(float)


experiment.run_if_main()
