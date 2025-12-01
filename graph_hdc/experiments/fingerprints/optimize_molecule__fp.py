"""
Fingerprint-based Molecular Optimization Experiment

This experiment extends the base optimize_molecule.py experiment to use
traditional molecular fingerprints (Morgan, RDKit, etc.) for molecular
representations.

The experiment uses RDKit's fingerprint generators to convert SMILES strings
into binary or count fingerprint vectors, which are then optimized via gradient
descent to reach target property values.

Key Features:
    - Support for multiple fingerprint types (Morgan, RDKit, Atom Pair, Torsion)
    - Configurable fingerprint size and radius
    - Fast computation (no GPU required)
    - Well-established molecular descriptors

Design Considerations:
    While fingerprints are discrete by nature, the gradient-based optimization
    operates in a continuous relaxation of the fingerprint space. The optimized
    representations may not correspond to valid molecular structures, but the
    closest matching molecules in the test set provide a proxy for achievable
    targets.

Usage:
    Run directly or create configuration YAML files:

    .. code-block:: yaml

        extend: optimize_molecule__fp.py
        parameters:
            DATASET_NAME: "aqsoldb"
            TARGET_INDEX: 0
            TARGET_VALUE: 5.0
            FINGERPRINT_TYPE: "morgan"
            FINGERPRINT_SIZE: 4096
            FINGERPRINT_RADIUS: 2
            ENSEMBLE_SIZE: 10
            OPTIMIZATION_EPOCHS: 200

Example:
    .. code-block:: bash

        # Run with debug mode
        python optimize_molecule__fp.py

        # Run with configuration
        python -m pycomex run optimize_molecule__fp__aqsoldb.yml
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict

# == FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_SIZE:
#       The size of the fingerprint vector to be generated. This determines the
#       dimensionality of the fingerprint representation. Common values are 1024,
#       2048, 4096, or 8192. Larger sizes can encode more structural information
#       and may provide smoother optimization landscapes.
FINGERPRINT_SIZE: int = 2048 * 2

# :param FINGERPRINT_RADIUS:
#       The radius parameter for Morgan/ECFP fingerprints. This determines how many
#       bonds away from each atom to consider when computing the circular substructure.
#       A radius of 2 corresponds to ECFP4, radius of 3 to ECFP6, etc.
#       Larger radii capture more context but may lead to sparser fingerprints.
FINGERPRINT_RADIUS: int = 2

# :param FINGERPRINT_TYPE:
#       The type of fingerprint to generate. Options include:
#       - 'morgan': Morgan/ECFP fingerprints (most common, good for similarity)
#       - 'rdkit': RDKit path-based fingerprints (based on linear paths)
#       - 'atom': Atom pair fingerprints (based on pairs of atoms and distances)
#       - 'torsion': Topological torsion fingerprints (based on 4-atom paths)
#
#       Morgan fingerprints are generally recommended for optimization tasks as they
#       provide a good balance between local and global structure representation.
FINGERPRINT_TYPE: str = 'morgan'

# == EXPERIMENT ==

experiment = Experiment.extend(
    'optimize_molecule.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                    index_data_map: dict[int, GraphDict]
                    ) -> None:
    """
    Process molecules into molecular fingerprint representations using RDKit.

    This hook converts SMILES strings directly into fingerprint vectors using
    RDKit's fingerprint generators. The type of fingerprint is controlled by
    the FINGERPRINT_TYPE parameter.

    Fingerprints provide a compact, fixed-size representation of molecular structure
    that is particularly well-suited for similarity calculations and machine learning.
    However, they are discrete by nature, so gradient-based optimization operates
    in a continuous relaxation of the fingerprint space.

    Optimization Considerations:
    - The optimization will modify fingerprint bits continuously (0 to 1)
    - Optimized representations may not correspond to valid molecules
    - Closest test set matching provides a practical validation strategy
    - Morgan fingerprints often work best due to their compositional nature

    Supported fingerprint types:
    - morgan: Circular fingerprints similar to ECFP
    - rdkit: Path-based fingerprints
    - atom: Atom pair fingerprints based on atom pairs and distances
    - torsion: Topological torsion fingerprints based on 4-atom paths

    :param e: The experiment instance providing access to parameters and logging.
    :param index_data_map: Dictionary mapping indices to graph dictionaries. This
        dictionary is modified in-place to add 'graph_features' to each entry.

    :return: None. Modifies index_data_map in-place by adding 'graph_features' key.
    """
    e.log('creating fingerprint generator...')

    # Select the appropriate fingerprint generator based on type
    if e.FINGERPRINT_TYPE == 'morgan':
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=e.FINGERPRINT_RADIUS,
            fpSize=e.FINGERPRINT_SIZE,
        )
        e.log(f'using Morgan fingerprints with radius={e.FINGERPRINT_RADIUS}, '
              f'size={e.FINGERPRINT_SIZE}')

    elif e.FINGERPRINT_TYPE == 'rdkit':
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=e.FINGERPRINT_SIZE,
            maxPath=2 * e.FINGERPRINT_RADIUS,
        )
        e.log(f'using RDKit fingerprints with maxPath={2*e.FINGERPRINT_RADIUS}, '
              f'size={e.FINGERPRINT_SIZE}')

    elif e.FINGERPRINT_TYPE == 'atom':
        gen = rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=e.FINGERPRINT_SIZE,
        )
        e.log(f'using Atom Pair fingerprints with size={e.FINGERPRINT_SIZE}')

    elif e.FINGERPRINT_TYPE == 'torsion':
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=e.FINGERPRINT_SIZE,
        )
        e.log(f'using Topological Torsion fingerprints with size={e.FINGERPRINT_SIZE}')

    else:
        raise ValueError(
            f"Unknown fingerprint type: {e.FINGERPRINT_TYPE}. "
            f"Supported types are: 'morgan', 'rdkit', 'atom', 'torsion'"
        )

    # Process each molecule to generate fingerprints
    e.log('processing molecules into fingerprints...')

    for c, (index, graph) in enumerate(index_data_map.items()):
        smiles: str = graph['graph_repr']
        mol = Chem.MolFromSmiles(smiles)

        # Generate fingerprint
        fingerprint = gen.GetFingerprint(mol)

        # Convert to numpy array and store
        # Note: Fingerprints are binary (0/1) but we convert to float for optimization
        graph['graph_features'] = np.array(fingerprint).astype(float)

        if c % 1000 == 0 and c > 0:
            e.log(f' * processed {c} molecules')

    e.log(f'completed fingerprint generation for {len(index_data_map)} molecules')


experiment.run_if_main()
