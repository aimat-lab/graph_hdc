import os
import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data.main import load_graph_dataset

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map

# == EXPERIMENT PARAMETERS ==

#: The name of the dataset to be used for the experiment.
DATASET_NAME: str = 'bace'

#: The dimensionality of the vector representations for both encoders.
VECTOR_SIZE: int = 2048

#: The depth of the hypernetwork encoder which corresponds to the Morgan radius
#: used for the fingerprint baseline.
DEPTH: int = 2

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def encode_hdc(smiles_list: List[str], dim: int, depth: int) -> Tuple[List[np.ndarray], float]:
    """Encode molecules using the HyperNet encoder.

    :param smiles_list: A list of SMILES strings representing the molecules.
    :param dim: The dimensionality of the resulting hypervectors.
    :param depth: The number of message passing steps of the encoder.

    :returns: A tuple containing the list of hypervectors and the total encoding
        time in seconds.
    """
    node_encoder_map = make_molecule_node_encoder_map(dim=dim)
    hyper_net = HyperNet(
        hidden_dim=dim,
        depth=depth,
        node_encoder_map=node_encoder_map,
    )

    graphs = [graph_dict_from_mol(Chem.MolFromSmiles(smi)) for smi in smiles_list]
    start = time.perf_counter()
    results = hyper_net.forward_graphs(graphs)
    end = time.perf_counter()

    hvs = [res['graph_embedding'] for res in results]
    return hvs, end - start


def encode_fingerprint(smiles_list: List[str], dim: int, radius: int) -> Tuple[List[np.ndarray], float]:
    """Encode molecules using RDKit Morgan fingerprints.

    :param smiles_list: A list of SMILES strings representing the molecules.
    :param dim: The dimensionality of the fingerprint vector.
    :param radius: The Morgan fingerprint radius.

    :returns: A tuple containing the list of fingerprints and the total encoding
        time in seconds.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=dim,
    )

    fingerprints = []
    start = time.perf_counter()
    for smi in smiles_list:
        fp = generator.GetFingerprint(Chem.MolFromSmiles(smi))
        fingerprints.append(np.array(fp).astype(float))
    end = time.perf_counter()

    return fingerprints, end - start


@experiment
def experiment(e: Experiment) -> None:
    """Run the runtime comparison experiment."""

    e.log(f'loading dataset "{e.DATASET_NAME}"...')
    graphs = load_graph_dataset(e.DATASET_NAME, folder_path='/tmp')
    smiles_list = [g['graph_repr'] for g in graphs]
    e.log(f'loaded {len(smiles_list)} molecules')

    e.log('encoding with HyperNet...')
    hdc_vectors, hdc_time = encode_hdc(smiles_list, e.VECTOR_SIZE, e.DEPTH)
    e['runtime/hdc_total'] = hdc_time
    e['runtime/hdc_average'] = hdc_time / len(hdc_vectors)

    e.log('encoding with Morgan fingerprints...')
    fp_vectors, fp_time = encode_fingerprint(smiles_list, e.VECTOR_SIZE, e.DEPTH)
    e['runtime/fp_total'] = fp_time
    e['runtime/fp_average'] = fp_time / len(fp_vectors)

    df = pd.DataFrame({
        'method': ['hdc', 'fingerprint'],
        'avg_time': [e['runtime/hdc_average'], e['runtime/fp_average']],
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='method', y='avg_time', ax=ax)
    ax.set_ylabel('Average Encoding Time [s]')
    ax.set_xlabel('Method')
    ax.set_title('Encoding Runtime Comparison')
    e.commit_fig('encoding_times.png', fig)
    e.commit_json('encoding_times.json', df.to_dict())


experiment.run_if_main()
