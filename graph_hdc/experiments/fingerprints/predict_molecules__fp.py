import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

FINGERPRINT_SIZE: int = 2048
FINGERPRINT_RADIUS: int = 2

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
    
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=e.FINGERPRINT_RADIUS, 
        fpSize=e.FINGERPRINT_SIZE,
    )
    
    for c, (index, graph) in enumerate(index_data_map.items()):
        smiles: str = graph['graph_repr']
        fingerprint = gen.GetFingerprint(Chem.MolFromSmiles(smiles))
        graph['graph_features'] = np.array(fingerprint).astype(float)

        if c % 1000 == 0:
            e.log(f' * {c} molecules done')

experiment.run_if_main()