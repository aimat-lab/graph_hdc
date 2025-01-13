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
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'classification'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = 0.1


# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__gnn.py',
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


@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict[int, dict],
                  **kwargs,
                  ) -> None:
    """
    This hook is executed after the dataset is loaded. It is used to perform any additional processing
    on the dataset before the experiment is run.
    ---
    In this case, we use the RDKit library to calculate the CLogP values for the molecules in the dataset, 
    since we are using a dataset which does not contain the labels directly.
    """
    e.log('calculating CLogP values and replacing targets...')
    
    for _, graph in index_data_map.items():
        smiles = str(graph['graph_repr'])
        mol = Chem.MolFromSmiles(smiles)
        
        conjugated_systems = find_conjungated_systems(mol)
        #pprint(conjugated_systems)
        atom_count = sum(len(system) for system in conjugated_systems)
        is_conjugated = int(atom_count == mol.GetNumAtoms())
        
        #graph['graph_labels'] = np.array([MolLogP(mol)])
        graph['graph_labels'] = np.array([1 - is_conjugated, is_conjugated]).astype(float)


experiment.run_if_main()