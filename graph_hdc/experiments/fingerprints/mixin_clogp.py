"""
Experiment mixin for CLogP calculation.

This mixin provides a hook to calculate CLogP values for molecules using RDKit
and replace the target values in the dataset.
"""
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.Crippen import MolLogP
from pycomex.functional.experiment import Experiment, ExperimentMixin

# Create the mixin instance
mixin = ExperimentMixin(glob=globals())


@mixin.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict[int, dict],
                  **kwargs,
                  ) -> None:
    """
    Calculate CLogP values for molecules and replace target labels.

    This hook is executed after the dataset is loaded. It uses the RDKit library
    to calculate the CLogP (calculated LogP) values for the molecules in the dataset,
    since we are using a dataset which does not contain these labels directly.

    :param e: The experiment instance providing logging and tracking functionality.
    :param index_data_map: Dictionary mapping indices to graph data dictionaries.
    :param kwargs: Additional keyword arguments (unused but required for hook signature).

    :return: None. Modifies the graph data in-place by updating 'graph_labels'.
    """
    e.log('calculating CLogP values and replacing targets...')

    for _, graph in index_data_map.items():
        smiles = str(graph['graph_repr'])
        mol = Chem.MolFromSmiles(smiles)
        graph['graph_labels'] = np.array([MolLogP(mol)])
