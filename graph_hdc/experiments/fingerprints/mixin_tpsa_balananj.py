"""
Experiment mixin for TPSA-BalabanJ synthetic property calculation.

This mixin provides functionality to calculate a synthetic molecular property that
combines two global molecular descriptors: the Balaban J index and the Topological
Polar Surface Area (TPSA). This property is specifically designed to require large
message passing radii in graph neural networks due to its dependence on global
molecular topology.

The TPSA-BalabanJ property is calculated as:
    TPSA-BalabanJ = BalabanJ × (1 + TPSA/100)

Why this property requires large message passing radius:

1. **BalabanJ Component**: The Balaban J index is a topological index that depends
   on the entire molecular graph's distance matrix and connectivity. It requires
   calculating a distance/adjacency matrix for the entire molecule and considers
   the global topology through the sum of distance matrix elements.

2. **TPSA Component**: The Topological Polar Surface Area (TPSA) is calculated as
   a sum of fragment-based contributions from across the entire molecule,
   specifically considering all polar atoms and their environments.

Both descriptors inherently capture global molecular properties, making them
ideal for testing the effectiveness of message passing networks with varying
neighborhood aggregation radii.
"""
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import GraphDescriptors
from pycomex.functional.experiment import Experiment, ExperimentMixin

# Create the mixin instance
mixin = ExperimentMixin(glob=globals())


@mixin.hook('get_graph_labels', replace=True, default=False)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    """
    Generate regression labels based on the TPSA-BalabanJ synthetic property.

    This hook replaces the default label extraction with a custom calculation that
    combines two global molecular descriptors: the Balaban J index (a topological
    descriptor based on distance matrices) and TPSA (Topological Polar Surface Area).

    The combined property is computed as:
        TPSA-BalabanJ = BalabanJ × (1 + TPSA/100)

    This property is designed to test graph neural networks' ability to capture
    global molecular features, as both components require information from the
    entire molecular structure rather than just local neighborhoods.

    Example:
        For a molecule with BalabanJ = 2.5 and TPSA = 50.0:
        TPSA-BalabanJ = 2.5 × (1 + 50.0/100) = 2.5 × 1.5 = 3.75

    :param e: The experiment instance providing logging and tracking functionality.
    :param index: The index of the current graph in the dataset.
    :param graph: Dictionary containing graph data including 'graph_repr' (SMILES string).

    :return: NumPy array of shape (1,) containing the TPSA-BalabanJ value.
    """
    smiles = str(graph['graph_repr'])
    mol = Chem.MolFromSmiles(smiles)

    # Calculate the Balaban J index (global topological descriptor)
    balaban_j = GraphDescriptors.BalabanJ(mol)

    # Calculate the Topological Polar Surface Area (global descriptor)
    tpsa = Descriptors.TPSA(mol)

    # Compute the combined TPSA-BalabanJ property
    tpsa_balananj = balaban_j * (1 + tpsa / 100)
    #print(tpsa_balananj)

    graph['graph_labels'] = np.array([tpsa_balananj]).astype(float)
    return graph['graph_labels']
