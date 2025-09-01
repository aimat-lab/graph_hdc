import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.Crippen import MolLogP
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'aqsoldb'
# :param DATASET_NAME_ID:
#       The name of the dataset to be used later on for the identification of the dataset. This name will NOT be used 
#       for the downloading of the dataset but only later on for identification. In most cases these will be the same 
#       but in cases for example one dataset is used as the basis of some deterministic calculation of the target values 
#       and in this case the name should identify it as such.
DATASET_NAME_ID: str = 'conjugated'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'binary'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = 0.1
DATASET_NOISE: float = 0.0

# == EMBEDDING PARAMETERS ==

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors. This will be the number of elements in each of the 
#       hypervectors that represent the individual molecular graphs.
EMBEDDING_SIZE: int = 2048 * 4
# :param NUM_LAYERS:
#       The number of layers in the hypernetwork. This parameter determines the depth of the hypernetwork
#       which is used to generate the graph embeddings. This means it is the number of message passing 
#       steps applied in the encoder.
NUM_LAYERS: int = 3

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__hdc.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

def find_conjungated_systems(mol: Chem.Mol) -> list:
    # Get bond conjugation info
    conjugated_bonds = [
        bond for bond in mol.GetBonds() if bond.GetIsConjugated()
    ]

    # Traverse the graph to group connected conjugated bonds
    conjugated_systems = []
    visited_bonds = set()

    for bond in conjugated_bonds:
        if bond.GetIdx() in visited_bonds:
            continue

        # Perform a DFS/BFS to identify all bonds in this conjugated system
        system = []
        stack = [bond]
        while stack:
            current_bond = stack.pop()
            if current_bond.GetIdx() in visited_bonds:
                continue
            visited_bonds.add(current_bond.GetIdx())
            system.append(current_bond)

            # Add connected conjugated bonds
            for neighbor in current_bond.GetBeginAtom().GetBonds():
                if neighbor.GetIsConjugated() and neighbor not in system:
                    stack.append(neighbor)
            for neighbor in current_bond.GetEndAtom().GetBonds():
                if neighbor.GetIsConjugated() and neighbor not in system:
                    stack.append(neighbor)

        conjugated_systems.append(system)

    return conjugated_systems


@experiment.hook('get_graph_labels', replace=True, default=False)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    smiles = str(graph['graph_repr'])
    mol = Chem.MolFromSmiles(smiles)
    
    conjugated_systems = find_conjungated_systems(mol)
    atom_count = sum(len(system) for system in conjugated_systems)
    is_conjugated = int(atom_count == mol.GetNumAtoms())
    
    graph['graph_labels'] = np.array([1 - is_conjugated, is_conjugated]).astype(float)
    return graph['graph_labels']


experiment.run_if_main()