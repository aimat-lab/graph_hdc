"""
Experiment mixin for Wiener Index calculation.

This mixin provides functionality to calculate the Wiener Index as a molecular
property for regression tasks. The Wiener Index is a classic topological
descriptor that measures the compactness of a molecular structure.

The Wiener Index is defined as the sum of all shortest path lengths between
pairs of atoms in a molecule:
    W = (1/2) × Σ(i,j) d(i,j)

where d(i,j) is the shortest path distance between atoms i and j in the
molecular graph.

Why this property requires large message passing radius:

1. **Global Topology**: The Wiener Index depends on distances between all pairs
   of atoms in the molecule, not just local neighborhoods. To accurately capture
   this property, a graph neural network needs to aggregate information from the
   entire molecular structure.

2. **Path-Dependent**: The index is calculated based on shortest paths through
   the molecular graph, which means that information about distant atoms must
   be propagated through multiple hops to correctly compute the descriptor.

3. **Structural Sensitivity**: Small changes in molecular topology (like adding
   a single bond or atom) can significantly affect the Wiener Index, as it
   influences many pairwise distances. This makes it a strong test for
   message passing networks' ability to capture global structural features.

The Wiener Index is widely used in QSPR/QSAR studies and has been shown to
correlate with various physical and chemical properties such as boiling points,
viscosity, and surface tension.
"""
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdmolops
from pycomex.functional.experiment import Experiment, ExperimentMixin

# Create the mixin instance
mixin = ExperimentMixin(glob=globals())


def calculate_wiener_index(mol: Chem.Mol) -> float:
    """
    Calculate the Wiener Index for a molecular structure.

    The Wiener Index is computed as the sum of all pairwise shortest path
    distances in the molecular graph. This implementation uses RDKit's
    distance matrix to efficiently compute all pairwise distances.

    :param mol: RDKit molecule object.

    :return: The Wiener Index as a float value.
    """
    # Get the distance matrix for all atom pairs
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    num_atoms = mol.GetNumAtoms()

    # Sum all pairwise distances (upper triangle to avoid double counting)
    wiener_index = 0.0
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            wiener_index += distance_matrix[i, j]

    return wiener_index


@mixin.hook('get_graph_labels', replace=True, default=False)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    """
    Generate regression labels based on the Wiener Index topological descriptor.

    This hook replaces the default label extraction with a custom calculation
    of the Wiener Index, which is the sum of all shortest path distances between
    pairs of atoms in the molecular graph.

    The Wiener Index is calculated manually using RDKit's distance matrix:
        W = Σ(i<j) d(i,j)

    where d(i,j) is the topological distance (number of bonds in shortest path)
    between atoms i and j.

    This property is designed to test graph neural networks' ability to capture
    global molecular topology, as it requires aggregating distance information
    from across the entire molecular structure.

    Example:
        For a linear chain of 4 carbon atoms (butane):
        - Distance pairs: (1,2)=1, (1,3)=2, (1,4)=3, (2,3)=1, (2,4)=2, (3,4)=1
        - Wiener Index = 1+2+3+1+2+1 = 10

    :param e: The experiment instance providing logging and tracking functionality.
    :param index: The index of the current graph in the dataset.
    :param graph: Dictionary containing graph data including 'graph_repr' (SMILES string).

    :return: NumPy array of shape (1,) containing the Wiener Index value.
    """
    smiles = str(graph['graph_repr'])
    mol = Chem.MolFromSmiles(smiles)

    # Calculate the Wiener Index manually using distance matrix
    # This computes the sum of all pairwise shortest path distances
    wiener_index = calculate_wiener_index(mol)

    # Store the calculated value as the regression target
    graph['graph_labels'] = np.array([wiener_index]).astype(float)
    return graph['graph_labels']
