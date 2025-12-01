"""
Fingerprint-based Molecular Representation Investigation

This experiment extends the base molecule_representation.py experiment to use
traditional molecular fingerprints (Morgan, RDKit, etc.) for molecular representations.

The experiment uses RDKit's fingerprint generators to convert SMILES strings into
binary or count fingerprint vectors.
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
#       2048, 4096, or 8192.
FINGERPRINT_SIZE: int = 2048 * 2

# :param FINGERPRINT_RADIUS:
#       The radius parameter for Morgan/ECFP fingerprints. This determines how many
#       bonds away from each atom to consider when computing the circular substructure.
#       A radius of 2 corresponds to ECFP4, radius of 3 to ECFP6, etc.
FINGERPRINT_RADIUS: int = 2

# :param FINGERPRINT_TYPE:
#       The type of fingerprint to generate. Options include:
#       - 'morgan': Morgan/ECFP fingerprints (most common)
#       - 'rdkit': RDKit path-based fingerprints
#       - 'atom': Atom pair fingerprints
#       - 'torsion': Topological torsion fingerprints
FINGERPRINT_TYPE: str = 'morgan'

# == UMAP PARAMETERS ==

# :param UMAP_METRIC:
#       Override the UMAP metric for fingerprints. Jaccard distance is particularly
#       well-suited for binary fingerprint vectors as it measures the similarity based
#       on the proportion of shared bits.
UMAP_METRIC: str = 'jaccard'

# == EXPERIMENT ==

experiment = Experiment.extend(
    'molecule_representation.py',
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
        graph['graph_features'] = np.array(fingerprint).astype(float)

        if c % 1000 == 0 and c > 0:
            e.log(f' * processed {c} molecules')

    e.log(f'completed fingerprint generation for {len(index_data_map)} molecules')


experiment.run_if_main()
