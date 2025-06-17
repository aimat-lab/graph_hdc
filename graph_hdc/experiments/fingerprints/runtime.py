import os
import time
from typing import Tuple, List, Dict, Any

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

GraphDict = Dict[str, Any]

# == EXPERIMENT PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to use. This will be used to download the dataset from the ChemMatData file share.
DATASET_NAME: str = 'clintox'
# :param VECTOR_SIZE:
#       The dimensionality of the resulting hypervectors / fingerprints.
VECTOR_SIZE: int = 2048
# :param DEPTH:
#       The number of message passing steps of the HyperNet encoder and at the same time 
#       the radius of the Morgan fingerprint.
DEPTH: int = 2
# :param USE_BATCHING:
#       Whether to use the batching feature of the HyperNet encoder or not.
USE_BATCHING: bool = True
# :param DEVICE:
#       The device to use for the HyperNet encoder.
DEVICE: str = 'cpu'

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('load_dataset', replace=False, default=True)
def load_dataset(e: Experiment) -> dict[int, GraphDict]:
    
    # This function will download the dataset from the ChemMatData file share and return the already pre-processed 
    # list of graph dict representations.
    graphs: list[GraphDict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path='/tmp'
    )
    
    index_data_map = dict(enumerate(graphs))
    return index_data_map


def encode_hdc(smiles_list: List[str], 
               dim: int, 
               depth: int, 
               use_batching: bool = True,
               device: str = 'cpu',
               batch_size: int = 8,
               ) -> Tuple[List[np.ndarray], float, List[float]]:
    """Encode molecules using the HyperNet encoder.

    :param smiles_list: A list of SMILES strings representing the molecules.
    :param dim: The dimensionality of the resulting hypervectors.
    :param depth: The number of message passing steps of the encoder.
    :param use_batching: Whether to use batching in HyperNet.forward_graphs.

    :returns: A tuple containing the list of hypervectors, the total encoding
        time in seconds, and a list of per-molecule encoding times.
    """
    node_encoder_map = make_molecule_node_encoder_map(dim=dim)
    hyper_net = HyperNet(
        hidden_dim=dim,
        depth=depth,
        node_encoder_map=node_encoder_map,
        device=device,
    )

    graphs = [graph_dict_from_mol(Chem.MolFromSmiles(smi)) for smi in smiles_list]
    hvs = []
    times = []
    if use_batching:
        total_time = 0.0
        times = []
        hvs = []
        _bs = 128
        for i in range(0, len(graphs), _bs):
            batch_graphs = graphs[i:i + _bs]
            start = time.perf_counter()
            results = hyper_net.forward_graphs(batch_graphs, batch_size=batch_size)
            end = time.perf_counter()
            batch_time = end - start
            total_time += batch_time
            # Assign the same batch_time / batch_size to each molecule in the batch
            times.extend([batch_time / len(batch_graphs)] * len(batch_graphs))
            hvs.extend([res['graph_embedding'] for res in results])
    else:
        for graph in graphs:
            start = time.perf_counter()
            res = hyper_net.forward_graphs([graph])[0]
            end = time.perf_counter()
            hvs.append(res['graph_embedding'])
            times.append(end - start)
        total_time = sum(times)
    return hvs, total_time, times


def encode_fingerprint(smiles_list: List[str], dim: int, radius: int) -> Tuple[List[np.ndarray], float, List[float]]:
    """Encode molecules using RDKit Morgan fingerprints.

    :param smiles_list: A list of SMILES strings representing the molecules.
    :param dim: The dimensionality of the fingerprint vector.
    :param radius: The Morgan fingerprint radius.

    :returns: A tuple containing the list of fingerprints, the total encoding
        time in seconds, and a list of per-molecule encoding times.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=dim,
    )

    fingerprints = []
    times = []
    for smi in smiles_list:
        start = time.perf_counter()
        fp = generator.GetFingerprint(Chem.MolFromSmiles(smi))
        end = time.perf_counter()
        fingerprints.append(np.array(fp).astype(float))
        times.append(end - start)
    total_time = sum(times)
    return fingerprints, total_time, times


@experiment
def experiment(e: Experiment) -> None:
    """Run the runtime comparison experiment."""

    # ~ Loading the dataset ~
    e.log(f'loading dataset "{e.DATASET_NAME}"...')
    index_data_map: dict[int, GraphDict] = e.apply_hook('load_dataset')
    graphs = list(index_data_map.values())
    
    smiles_list = [g['graph_repr'] for g in graphs]
    
    e.log(f'loaded {len(smiles_list)} molecules')

    # ~ Encoding the molecules ~
    e.log('encoding with HyperNet...')
    hdc_vectors, hdc_time, hdc_times = encode_hdc(
        smiles_list, 
        dim=e.VECTOR_SIZE, 
        depth=e.DEPTH, 
        use_batching=e.USE_BATCHING, 
        device=e.DEVICE
    )
    e['runtime/hdc/total'] = hdc_time
    e['runtime/hdc/avg'] = hdc_time / len(hdc_vectors)
    e['runtime/hdc/std'] = np.std(hdc_times)

    e.log('encoding with Morgan fingerprints...')
    fp_vectors, fp_time, fp_times = encode_fingerprint(smiles_list, e.VECTOR_SIZE, e.DEPTH)
    e['runtime/fp/total'] = fp_time
    e['runtime/fp/avg'] = fp_time / len(fp_vectors)
    e['runtime/fp/std'] = np.std(fp_times)

    # Calculate speedup factor
    speedup = e['runtime/hdc/avg'] / e['runtime/fp/avg'] if e['runtime/fp/avg'] > 0 else float('inf')
    e['factor'] = speedup

    # Print results to console
    print("=== Encoding Runtime Results ===")
    print(f"HyperNet: total={e['runtime/hdc/total']:.6f}s, avg={e['runtime/hdc/avg']:.6f}s, std={e['runtime/hdc/std']:.6f}s")
    print(f"Fingerprint: total={e['runtime/fp/total']:.6f}s, avg={e['runtime/fp/avg']:.6f}s, std={e['runtime/fp/std']:.6f}s")
    print(f"Fingerprints are {speedup:.2f}x faster than HyperNet (avg time)")

    # Compute percentiles for error bars (ensure non-negative)
    hdc_q25, hdc_q75 = np.percentile(hdc_times, [25, 75])
    fp_q25, fp_q75 = np.percentile(fp_times, [25, 75])
    hdc_err_low = max(e['runtime/hdc/avg'] - hdc_q25, 0)
    hdc_err_high = max(hdc_q75 - e['runtime/hdc/avg'], 0)
    fp_err_low = max(e['runtime/fp/avg'] - fp_q25, 0)
    fp_err_high = max(fp_q75 - e['runtime/fp/avg'], 0)
    # error bars as (2, n) array: [[lower_errors], [upper_errors]]
    yerr = np.array([[hdc_err_low, fp_err_low], [hdc_err_high, fp_err_high]])

    df = pd.DataFrame({
        'method': ['hdc', 'fingerprint'],
        'avg_time': [e['runtime/hdc/avg'], e['runtime/fp/avg']],
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    bar = sns.barplot(data=df, x='method', y='avg_time', ax=ax, ci=None, color='#9cffcf')
    # Add percentile-based error bars manually
    ax.errorbar(
        x=[0, 1],
        y=df['avg_time'],
        yerr=yerr,
        fmt='none',
        c='black',
        capsize=5,
        linewidth=2,
    )
    ax.set_ylabel('Average Encoding Time [s]')
    ax.set_xlabel('Method')
    ax.set_title(f'Encoding Runtime Comparison\n'
                 f'HyperNet: {e["runtime/hdc/avg"]:.3f}s - Fingerprint: {e["runtime/fp/avg"]:.3f}s')
    e.commit_fig('encoding_times.pdf', fig)

    # Add violin plot for per-molecule times, suppressing outliers
    
    violin_df = pd.DataFrame({
        'time': hdc_times + fp_times,
        'method': ['hdc'] * len(hdc_times) + ['fingerprint'] * len(fp_times)
    })
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=violin_df, x='method', y='time', ax=ax2, bw_method='silverman', color='#9cffcf',)
    ax2.set_ylabel('Encoding Time per Molecule [s]')
    ax2.set_xlabel('Method')
    ax2.set_title('Encoding Time Distribution')
    e.commit_fig('encoding_times_violin.pdf', fig2)

    e.commit_json('encoding_times.json', df.to_dict())


experiment.run_if_main()
