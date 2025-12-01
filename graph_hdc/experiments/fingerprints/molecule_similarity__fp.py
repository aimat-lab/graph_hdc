"""
Molecular Similarity Experiment with Fingerprint Encoding

This module extends the molecule_similarity.py base experiment to use traditional
molecular fingerprints for encoding. Molecules are encoded into binary or count
fingerprints using RDKit's fingerprint generators, and similarity is computed
using Tanimoto (Jaccard) distance.

Key Features:
    - Multiple fingerprint types: Morgan, RDKit, Atom Pair, Topological Torsion
    - Configurable fingerprint size and parameters
    - Tanimoto distance metric for similarity computation
    - Fast computation using RDKit's optimized implementations

Usage:
    Create a YAML configuration file to run this experiment:

    .. code-block:: yaml

        extend: molecule_similarity__fp.py
        parameters:
            DATASET_NAME: "qm9_smiles"
            NUM_SAMPLES: 10
            NUM_NEIGHBORS: 5
            FINGERPRINT_TYPE: "morgan"
            FINGERPRINT_SIZE: 2048
            FINGERPRINT_RADIUS: 2
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==
# These are inherited from the base experiment but can be overridden

DATASET_NAME: str = 'aqsoldb'
DATASET_NAME_ID: str = DATASET_NAME

# == FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_TYPE:
#       The type of molecular fingerprint to generate. Options:
#       - 'morgan': Morgan (circular) fingerprints (ECFP-like)
#       - 'rdkit': RDKit path-based fingerprints
#       - 'atom_pair': Atom pair fingerprints
#       - 'torsion': Topological torsion fingerprints
FINGERPRINT_TYPE: str = 'morgan'

# :param FINGERPRINT_SIZE:
#       The size (number of bits) of the fingerprint. Larger sizes reduce
#       collision probability but increase memory usage. Common values are
#       1024, 2048, or 4096.
FINGERPRINT_SIZE: int = 2048

# :param FINGERPRINT_RADIUS:
#       The radius parameter for Morgan fingerprints (equivalent to ECFP diameter/2).
#       For example, radius=2 corresponds to ECFP4. This parameter is ignored for
#       non-Morgan fingerprint types. Common values: 2 or 3.
FINGERPRINT_RADIUS: int = 2

# :param USE_COUNTS:
#       Whether to use count fingerprints (True) or binary fingerprints (False).
#       Count fingerprints record the frequency of features, while binary fingerprints
#       only record presence/absence. Binary fingerprints are more common for
#       similarity searches.
USE_COUNTS: bool = False

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'molecule_similarity.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                   index_data_map: dict
                   ) -> None:
    """
    Process the dataset using RDKit to generate molecular fingerprints.

    This hook replaces the base implementation and uses RDKit's fingerprint
    generators to convert molecules into fingerprint vectors. The process
    involves:
        1. Creating a fingerprint generator based on FINGERPRINT_TYPE
        2. Processing all SMILES strings to generate fingerprints
        3. Converting fingerprints to numpy arrays
        4. Storing the resulting fingerprints as 'graph_features'

    :param e: The experiment instance.
    :param index_data_map: Dictionary to be modified in-place with 'graph_features'.

    :return: None. Modifies index_data_map in-place.
    """
    e.log('creating fingerprint generator...')
    e.log(f' * FINGERPRINT_TYPE: {e.FINGERPRINT_TYPE}')
    e.log(f' * FINGERPRINT_SIZE: {e.FINGERPRINT_SIZE}')

    # Create fingerprint generator based on type
    if e.FINGERPRINT_TYPE == 'morgan':
        e.log(f' * FINGERPRINT_RADIUS: {e.FINGERPRINT_RADIUS}')
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=e.FINGERPRINT_RADIUS,
            fpSize=e.FINGERPRINT_SIZE,
        )

    elif e.FINGERPRINT_TYPE == 'rdkit':
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=e.FINGERPRINT_SIZE,
        )

    elif e.FINGERPRINT_TYPE == 'atom_pair':
        gen = rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=e.FINGERPRINT_SIZE,
        )

    elif e.FINGERPRINT_TYPE == 'torsion':
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=e.FINGERPRINT_SIZE,
        )

    else:
        raise ValueError(
            f"Unknown FINGERPRINT_TYPE: '{e.FINGERPRINT_TYPE}'. "
            f"Supported types: 'morgan', 'rdkit', 'atom_pair', 'torsion'"
        )

    # Store generator as private instance attribute (NOT in serializable storage)
    # This prevents JSON serialization errors and ensures proper access
    e._fp_generator = gen

    # Generate fingerprints for all molecules
    e.log('generating fingerprints...')
    for c, (index, graph) in enumerate(index_data_map.items()):
        smiles: str = graph['graph_repr']
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            e.log(f'WARNING: Could not parse SMILES at index {index}: {smiles}')
            # Use zero vector as fallback
            graph['graph_features'] = np.zeros(e.FINGERPRINT_SIZE, dtype=float)
            continue

        # Generate fingerprint
        if e.USE_COUNTS:
            fingerprint = gen.GetCountFingerprint(mol)
        else:
            fingerprint = gen.GetFingerprint(mol)

        # Convert to numpy array
        graph['graph_features'] = np.array(fingerprint, dtype=float)

        if c % 1000 == 0 and c > 0:
            e.log(f' * {c} fingerprints generated')

    e.log(f'generated fingerprints for {len(index_data_map)} molecules')


@experiment.hook('encode_molecule', replace=True, default=False)
def encode_molecule(e: Experiment,
                   smiles: str
                   ) -> np.ndarray:
    """
    Encode a single molecule using the existing fingerprint generator.

    This hook reuses the fingerprint generator that was created during the initial
    process_dataset call, avoiding re-initialization. The generator is retrieved
    from e._fp_generator (private instance attribute).

    :param e: The experiment instance.
    :param smiles: SMILES string of the molecule to encode.

    :return: Fingerprint vector as numpy array, or None if encoding fails.
    """
    # Retrieve the stored fingerprint generator from private instance attribute
    gen = getattr(e, '_fp_generator', None)
    if gen is None:
        raise RuntimeError(
            "Fingerprint generator not found. Make sure process_dataset was called first."
        )

    try:
        # Convert SMILES to RDKit mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate fingerprint
        if e.USE_COUNTS:
            fingerprint = gen.GetCountFingerprint(mol)
        else:
            fingerprint = gen.GetFingerprint(mol)

        # Convert to numpy array
        return np.array(fingerprint, dtype=float)

    except Exception as ex:
        # Log error but don't crash - return None to indicate failure
        return None


@experiment.hook('compute_distance', replace=True, default=False)
def compute_distance(e: Experiment,
                    features1: np.ndarray,
                    features2: np.ndarray
                    ) -> float:
    """
    Compute Tanimoto (Jaccard) distance between two fingerprints.

    The Tanimoto coefficient (also known as Jaccard similarity for binary data)
    is a common similarity metric for molecular fingerprints:

        Tanimoto(A, B) = |A ∩ B| / |A ∪ B|

    For binary fingerprints:
        Tanimoto(A, B) = (A · B) / (|A|² + |B|² - A · B)

    The Tanimoto distance is defined as:
        Tanimoto_distance = 1 - Tanimoto_similarity

    This metric ranges from 0 (identical fingerprints) to 1 (no overlap).

    :param e: The experiment instance.
    :param features1: First fingerprint vector.
    :param features2: Second fingerprint vector.

    :return: Tanimoto distance (lower = more similar).
    """
    # Compute intersection (dot product for binary/count vectors)
    intersection = np.dot(features1, features2)

    # Compute union
    # For binary fingerprints: |A ∪ B| = |A| + |B| - |A ∩ B|
    # This also works for count fingerprints
    norm1_sq = np.dot(features1, features1)
    norm2_sq = np.dot(features2, features2)
    union = norm1_sq + norm2_sq - intersection

    # Avoid division by zero
    if union == 0:
        # Both fingerprints are zero vectors
        return 0.0 if norm1_sq == 0 and norm2_sq == 0 else 1.0

    # Compute Tanimoto similarity
    tanimoto_similarity = intersection / union

    # Convert to distance (0 = identical, 1 = no overlap)
    tanimoto_distance = 1 - tanimoto_similarity

    return float(tanimoto_distance)


experiment.run_if_main()
