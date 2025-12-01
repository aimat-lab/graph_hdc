"""
Experiment mixin for conjugated system detection.

This mixin provides functionality to detect whether a molecule is fully conjugated
by analyzing its bond structure using RDKit. It replaces the default label extraction
with a custom binary classification based on conjugation.
"""
import numpy as np
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment, ExperimentMixin

# Create the mixin instance
mixin = ExperimentMixin(glob=globals())


def find_conjungated_systems(mol: Chem.Mol) -> list:
    """
    Identify all conjugated systems within a molecule.

    This function traverses the molecular graph to identify groups of connected
    conjugated bonds. A conjugated system is a set of adjacent atoms connected by
    alternating single and multiple bonds, allowing electron delocalization.

    The algorithm uses depth-first search to group connected conjugated bonds into
    separate systems.

    :param mol: RDKit Mol object representing the molecule.

    :return: List of conjugated systems, where each system is a list of Bond objects.
    """
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


@mixin.hook('get_graph_labels', replace=True, default=False)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    """
    Generate binary classification labels based on molecular conjugation.

    This hook replaces the default label extraction with a custom classification
    that determines whether a molecule is fully conjugated. A molecule is considered
    fully conjugated if all its atoms participate in conjugated systems.

    The classification is binary:
    - [1, 0]: Not fully conjugated
    - [0, 1]: Fully conjugated

    :param e: The experiment instance providing logging and tracking functionality.
    :param index: The index of the current graph in the dataset.
    :param graph: Dictionary containing graph data including 'graph_repr' (SMILES string).

    :return: NumPy array of shape (2,) containing the binary classification labels.
    """
    smiles = str(graph['graph_repr'])
    mol = Chem.MolFromSmiles(smiles)

    conjugated_systems = find_conjungated_systems(mol)
    atom_count = sum(len(system) for system in conjugated_systems)
    is_conjugated = int(atom_count == mol.GetNumAtoms())

    graph['graph_labels'] = np.array([1 - is_conjugated, is_conjugated]).astype(float)
    return graph['graph_labels']
