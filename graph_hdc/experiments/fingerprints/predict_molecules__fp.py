import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_SIZE:
#       The size of the fingerprint to be generated. This will be the number of elements in the 
#       fingerprint vector representation of each molecule.
FINGERPRINT_SIZE: int = 2048
# :param FINGERPRINT_RADIUS:
#       The radius of the fingerprint to be generated. This parameter determines the number of
#       bonds to be considered when generating the fingerprint.
FINGERPRINT_RADIUS: int = 2
# :param FINGERPRINT_TYPE:
#       The type of fingerprint to be generated. This can be 'morgan' for Morgan fingerprints 
#       or 'rdkit' for RDKit fingerprints. The default is 'morgan'.
FINGERPRINT_TYPE: str = 'morgan' # rdkit, atom, torsion

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
    
    for c, (index, graph) in enumerate(index_data_map.items()):
        smiles: str = graph['graph_repr']
        fingerprint = gen.GetFingerprint(Chem.MolFromSmiles(smiles))
        graph['graph_features'] = np.array(fingerprint).astype(float)

        if c % 1000 == 0:
            e.log(f' * {c} molecules done')

experiment.run_if_main()