"""
Similarity-Based Bioactivity Prediction with Fingerprint Encoding

This module extends the predict_bioactivity.py base experiment to use traditional
molecular fingerprints for encoding. It supports multiple fingerprint types including
ECFP4 (Extended Connectivity Fingerprints) and MACCS keys, which serve as standard
baselines for comparison with HDC representations.

Fingerprints are binary or count vectors that encode structural features of molecules.
Similarity is computed using Tanimoto distance, the standard metric for fingerprints
in cheminformatics.

Supported Fingerprint Types:
    - ecfp4: Extended Connectivity Fingerprints (Morgan, radius=2, 2048 bits)
        - Standard baseline, typically achieves AUC ~0.75-0.85 for virtual screening
        - Encodes circular neighborhoods up to 2 bonds from each atom
    - ecfp6: Extended Connectivity Fingerprints (Morgan, radius=3, 2048 bits)
        - Larger radius captures more extended structural features
    - maccs: MACCS Structural Keys (166 bits)
        - Predefined substructural features based on medicinal chemistry knowledge
        - Fixed-size, highly interpretable
    - morgan: Morgan Fingerprints (radius=2, 2048 bits, count-based)
        - Count-based version of ECFP4

Key Features:
    - Multiple fingerprint types via FP_TYPE parameter
    - Tanimoto distance metric (standard for binary fingerprints)
    - Fast encoding without GPU requirements
    - Well-established baselines for benchmarking

Usage:
    Create a YAML configuration file to run this experiment:

    .. code-block:: yaml

        extend: predict_bioactivity__fp.py
        parameters:
            DATASET_NAME: "bl_chembl_reg"
            FP_TYPE: "ecfp4"
            NUM_QUERY_ACTIVES: 5
            NUM_REPETITIONS: 50
            SEED: 1

Expected Performance:
    - ECFP4: AUC ~0.75-0.85, BEDROC ~0.15-0.25
    - MACCS: AUC ~0.70-0.80, BEDROC ~0.12-0.20
    - Performance varies by target and dataset
"""
import time
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==
# These are inherited from the base experiment but can be overridden

DATASET_NAME: str = 'bl_chembl_reg'
DATASET_NAME_ID: str = DATASET_NAME

# == FINGERPRINT PARAMETERS ==

# :param FP_TYPE:
#       The type of molecular fingerprint to use for encoding. Options:
#       - 'ecfp4': Extended Connectivity Fingerprints radius=2 (2048 bits)
#           Standard baseline, achieves AUC ~0.75-0.85
#       - 'ecfp6': Extended Connectivity Fingerprints radius=3 (2048 bits)
#           Larger radius for more extended features
#       - 'maccs': MACCS Structural Keys (166 bits)
#           Predefined substructural features
#       - 'morgan': Morgan Fingerprints radius=2 (2048 bits, count-based)
#           Count-based version of ECFP4
FP_TYPE: Literal['ecfp4', 'ecfp6', 'maccs', 'morgan'] = 'ecfp4'

# :param FP_SIZE:
#       The size (number of bits) for the fingerprint. Only applicable for
#       Morgan/ECFP fingerprints. MACCS keys have fixed size of 166 bits.
#       Common values: 1024, 2048, 4096.
FP_SIZE: int = 2048

# :param FP_RADIUS:
#       The radius parameter for Morgan/ECFP fingerprints. This determines the
#       maximum number of bonds from each atom to include in circular neighborhoods.
#       - radius=2: ECFP4 (standard)
#       - radius=3: ECFP6 (extended)
#       Not applicable for MACCS keys.
FP_RADIUS: int = 2

# :param USE_COUNTS:
#       Whether to use count-based fingerprints (True) or binary fingerprints (False).
#       Count-based fingerprints preserve information about feature frequency but
#       may be less robust. Only applicable for Morgan/ECFP fingerprints.
USE_COUNTS: bool = False

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_bioactivity.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(
    e: Experiment,
    index_data_map: dict
) -> None:
    """
    Process the bioactivity dataset using molecular fingerprints.

    This hook replaces the base implementation and computes fingerprints for
    all molecules in the dataset. The fingerprints are stored as 'graph_features'
    in numpy array format.

    Supported fingerprint types:
        - ecfp4/ecfp6: Morgan fingerprints (circular connectivity)
        - maccs: MACCS structural keys (predefined substructures)
        - morgan: Count-based Morgan fingerprints

    The encoding is fast and doesn't require GPU acceleration, making it suitable
    for large-scale virtual screening benchmarks.

    :param e: The experiment instance.
    :param index_data_map: Dictionary to be modified in-place with 'graph_features'.

    :return: None. Modifies index_data_map in-place.
    """
    e.log(f'processing molecules into {e.FP_TYPE.upper()} fingerprints...')

    if e.FP_TYPE in ['ecfp4', 'ecfp6', 'morgan']:
        # Determine radius based on FP_TYPE if not explicitly set
        if e.FP_TYPE == 'ecfp4':
            radius = 2
        elif e.FP_TYPE == 'ecfp6':
            radius = 3
        else:
            radius = e.FP_RADIUS

        e.log(f' * fingerprint type: Morgan (ECFP)')
        e.log(f' * radius: {radius}')
        e.log(f' * size: {e.FP_SIZE} bits')
        e.log(f' * use_counts: {e.USE_COUNTS}')

    elif e.FP_TYPE == 'maccs':
        e.log(f' * fingerprint type: MACCS keys')
        e.log(f' * size: 166 bits (fixed)')

    time_start = time.time()

    for c, (index, data) in enumerate(index_data_map.items()):
        smiles: str = data['graph_repr']
        mol: Chem.Mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # This shouldn't happen after filtering, but handle gracefully
            e.log(f'WARNING: Could not parse SMILES for index {index}: {smiles}')
            # Create zero vector as placeholder
            if e.FP_TYPE == 'maccs':
                index_data_map[index]['graph_features'] = np.zeros(166, dtype=np.float32)
            else:
                index_data_map[index]['graph_features'] = np.zeros(e.FP_SIZE, dtype=np.float32)
            continue

        # Generate fingerprint based on type
        if e.FP_TYPE in ['ecfp4', 'ecfp6', 'morgan']:
            # Use rdFingerprintGenerator for modern RDKit versions
            if e.FP_TYPE == 'ecfp4':
                radius = 2
            elif e.FP_TYPE == 'ecfp6':
                radius = 3
            else:
                radius = e.FP_RADIUS

            # Generate Morgan fingerprint
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=e.FP_SIZE
            )

            if e.USE_COUNTS:
                # Count-based fingerprint
                fingerprint = gen.GetCountFingerprint(mol)
                # Convert to numpy array (counts)
                fp_array = np.zeros(e.FP_SIZE, dtype=np.float32)
                for idx, count in fingerprint.GetNonzeroElements().items():
                    fp_array[idx] = count
            else:
                # Binary fingerprint
                fingerprint = gen.GetFingerprint(mol)
                # Convert to numpy array
                fp_array = np.array(fingerprint, dtype=np.float32)

        elif e.FP_TYPE == 'maccs':
            # MACCS keys (166 bits)
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
            # Convert to numpy array
            fp_array = np.array(fingerprint, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported fingerprint type: {e.FP_TYPE}")

        # Store fingerprint as graph_features
        index_data_map[index]['graph_features'] = fp_array

        if c % 1000 == 0 and c > 0:
            e.log(f' * {c} molecules processed')

    time_end = time.time()
    e.log(f'processed {len(index_data_map)} molecules into fingerprints after '
          f'{time_end - time_start:.2f} seconds')

    # Log fingerprint statistics
    fp_arrays = [data['graph_features'] for data in index_data_map.values()]
    if len(fp_arrays) > 0:
        if e.USE_COUNTS:
            # For count fingerprints, report sparsity
            nonzero_counts = [np.count_nonzero(fp) for fp in fp_arrays]
            avg_nonzero = np.mean(nonzero_counts)
            sparsity = 1.0 - (avg_nonzero / fp_arrays[0].shape[0])
            e.log(f' * average nonzero features: {avg_nonzero:.1f}')
            e.log(f' * sparsity: {sparsity:.3f}')
        else:
            # For binary fingerprints, report average bit density
            bit_densities = [np.mean(fp) for fp in fp_arrays]
            avg_density = np.mean(bit_densities)
            e.log(f' * average bit density: {avg_density:.3f}')


@experiment.hook('compute_distance', replace=True, default=False)
def compute_distance(e: Experiment,
                    features1: np.ndarray,
                    features2: np.ndarray
                    ) -> float:
    """
    Compute Tanimoto distance between two fingerprints.

    Tanimoto distance is the standard similarity metric for molecular fingerprints
    in cheminformatics. It is also known as Jaccard distance and measures the
    overlap between two sets of features.

    For binary fingerprints, Tanimoto similarity is defined as:
        T(A, B) = |A ∩ B| / |A ∪ B|
        T(A, B) = (A · B) / (||A||₁ + ||B||₁ - A · B)

    Where:
        - A · B is the number of bits set to 1 in both fingerprints
        - ||A||₁ is the number of bits set to 1 in fingerprint A

    Tanimoto distance is then:
        distance = 1 - T(A, B)

    This metric ranges from 0 (identical fingerprints) to 1 (no overlap).

    For count-based fingerprints, the formula generalizes naturally.

    :param e: The experiment instance.
    :param features1: First fingerprint vector (binary or count-based).
    :param features2: Second fingerprint vector.

    :return: Tanimoto distance in [0, 1]. Lower values indicate more similar molecules.

    Example:

    .. code-block:: python

        # Two identical fingerprints have distance 0
        fp1 = np.array([1, 0, 1, 0, 1])
        fp2 = np.array([1, 0, 1, 0, 1])
        dist = compute_distance(e, fp1, fp2)  # Returns 0.0

        # No overlap gives distance 1
        fp1 = np.array([1, 0, 0, 0])
        fp2 = np.array([0, 1, 1, 1])
        dist = compute_distance(e, fp1, fp2)  # Returns 1.0
    """
    # Compute dot product (intersection)
    dot_product = np.dot(features1, features2)

    # Compute norms (sum of features)
    norm1 = np.sum(features1)
    norm2 = np.sum(features2)

    # Compute union
    union = norm1 + norm2 - dot_product

    # Avoid division by zero
    if union == 0:
        # Both vectors are all zeros
        return 0.0 if norm1 == 0 and norm2 == 0 else 1.0

    # Compute Tanimoto similarity
    tanimoto_similarity = dot_product / union

    # Convert to distance
    # Clip to [0, 1] to handle numerical precision issues
    tanimoto_distance = np.clip(1.0 - tanimoto_similarity, 0.0, 1.0)

    return float(tanimoto_distance)


experiment.run_if_main()
