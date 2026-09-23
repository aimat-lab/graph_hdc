import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.baselines.minhash_fps import secfp_fingerprint, map4_fingerprint

# == FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_SIZE:
#       The size of the fingerprint to be generated. This will be the number of elements in the 
#       fingerprint vector representation of each molecule.
FINGERPRINT_SIZE: int = 2048
# :param FINGERPRINT_RADIUS:
#       The radius of the fingerprint to be generated. This parameter determines the number of
#       bonds to be considered when generating the fingerprint. For 'secfp' it is the maximum radius of
#       the circular substructures (3 = SECFP6, the MHFP default), for 'map4' the maximum radius of the
#       atom environments in the atom pairs (2 = MAP4).
FINGERPRINT_RADIUS: int = 2
# :param FINGERPRINT_TYPE:
#       The type of fingerprint to be generated: 'morgan', 'count_morgan', 'rdkit', 'atom' (atom pair),
#       'torsion' (topological torsion), 'secfp' (folded MHFP, Probst & Reymond 2018) or 'map4' (folded
#       MinHashed atom-pair fingerprint, Capecchi et al. 2020). See graph_hdc.baselines.minhash_fps for
#       the latter two.
FINGERPRINT_TYPE: str = 'morgan'

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                    index_data_map: dict
                    ) -> None:
    
    if e.FINGERPRINT_TYPE == 'morgan':
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=e.FINGERPRINT_RADIUS, 
            fpSize=e.FINGERPRINT_SIZE,
        )
        
    elif e.FINGERPRINT_TYPE == 'rdkit':
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=e.FINGERPRINT_SIZE,
            maxPath=2*e.FINGERPRINT_RADIUS,
        )
        
    elif e.FINGERPRINT_TYPE == 'atom':
        gen = rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=e.FINGERPRINT_SIZE,
        )
        
    elif e.FINGERPRINT_TYPE == 'torsion':
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=e.FINGERPRINT_SIZE,
        )

    elif e.FINGERPRINT_TYPE == 'count_morgan':
        # Count-based Morgan: identical substructure hashing to 'morgan', but the output vector holds
        # the number of times each bit is set (integer counts) instead of a 0/1 presence indicator.
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=e.FINGERPRINT_RADIUS,
            fpSize=e.FINGERPRINT_SIZE,
        )

    elif e.FINGERPRINT_TYPE == 'secfp':
        gen = None
        encode = lambda mol: secfp_fingerprint(mol, length=e.FINGERPRINT_SIZE, radius=e.FINGERPRINT_RADIUS)

    elif e.FINGERPRINT_TYPE == 'map4':
        gen = None
        encode = lambda mol: map4_fingerprint(mol, length=e.FINGERPRINT_SIZE, radius=e.FINGERPRINT_RADIUS)

    else:
        raise ValueError(f'unknown FINGERPRINT_TYPE "{e.FINGERPRINT_TYPE}"')

    count = (e.FINGERPRINT_TYPE == 'count_morgan')
    e.log(f'processing molecules into {"count " if count else ""}{e.FINGERPRINT_TYPE} fingerprints...')

    for c, (index, graph) in enumerate(index_data_map.items()):
        mol = Chem.MolFromSmiles(graph['graph_repr'])
        if gen is None:
            graph['graph_features'] = encode(mol).astype(float)
        elif count:
            graph['graph_features'] = np.array(gen.GetCountFingerprint(mol).ToList()).astype(float)
        else:
            graph['graph_features'] = np.array(gen.GetFingerprint(mol)).astype(float)

        if c % 1000 == 0:
            e.log(f' * {c} molecules done')

experiment.run_if_main()