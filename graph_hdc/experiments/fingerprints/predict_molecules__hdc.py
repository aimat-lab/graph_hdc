import os
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Literal, Dict, Optional

import umap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data.processing import MoleculeProcessing, OneHotEncoder
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdmolops

# from visual_graph_datasets.data import nx_from_graph
from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.utils import quantize_equal_frequency
from graph_hdc.special.spectrum import effective_components
from graph_hdc.special.molecules import (
    make_molecule_node_encoder_map,
    make_molecule_node_encoder_map_cont,
)
from graph_hdc.special.molecules import (
    make_molecule_graph_encoder_map_cont,
    make_value_graph_encoder,
)
import pandas as pd

DATASET_NAME: str = 'aqsoldb'
# :param DATASET_NAME_ID:
#       The name of the dataset to be used later on for the identification of the dataset. This name will NOT be used 
#       for the downloading of the dataset but only later on for identification. In most cases these will be the same 
#       but in cases for example one dataset is used as the basis of some deterministic calculation of the target values 
#       and in this case the name should identify it as such.
DATASET_NAME_ID: str = DATASET_NAME
DATASET_TYPE: str = 'regression'

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors. This will be the number of elements in each of the 
#       hypervectors that represent the individual molecular graphs.
EMBEDDING_SIZE: int = 2048
# :param NUM_LAYERS:
#       The number of layers in the hypernetwork. This parameter determines the depth of the hypernetwork
#       which is used to generate the graph embeddings. This means it is the number of message passing 
#       steps applied in the encoder.
NUM_LAYERS: int = 2
# :param BATCH_SIZE:
#       The size of the batches to be used during training. This parameter determines the number of samples
#       that are processed in parallel during the training of the model.
BATCH_SIZE: int = 8
# :param DEVICE:
#       The device to be used for the training of the model. This parameter can be set to 'cuda:0' to use the
#       GPU for training, or to 'cpu' to use the CPU.
#DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE: str = "cpu"
# :param ECODING_MODE:
#       This string determines the mode in which the HyperNet encoder operates in. The categorical mode
#       only uses categorical encodings for the node and graph features. The continuous mode is the newer 
#       version of the encoder that encodes certain features with the FHRR continuous encodings. 
ENCODING_MODE: Literal['categorical', 'continuous'] = 'continuous'
# :param BIDIRECTIONAL:
#       Whether messages are passed along both directions of every bond. The graph dicts created by
#       ``graph_dict_from_mol`` store each bond only once, so with False every atom only aggregates the
#       neighbors with a higher atom index and the fingerprint depends on the atom order of the input SMILES.
#       True implements the message passing as described in the paper (sum over all neighbors), which is
#       invariant to the atom order. Embedding caches of bidirectional runs carry a "__bidir" suffix.
BIDIRECTIONAL: bool = True
# :param HYDROGEN_COUNT:
#       How the hydrogen count of each atom is determined. "total" counts all bonded hydrogens (RDKit
#       GetTotalNumHs), as described in the paper. "implicit" (the behavior before this parameter existed) only
#       counts implicit hydrogens, which is 0 for every atom written in brackets ([nH], [NH3+], [C@@H], ...).
#       Embedding caches of "total" runs carry a "__totalh" suffix.
HYDROGEN_COUNT: str = 'total'
# :param STEREO:
#       Whether the atom attributes include the CIP stereo descriptor of every atom (R/S, r/s for stereocenters,
#       E/Z for both atoms of a stereogenic double bond; see graph_hdc.special.molecules.stereo_labels). Atoms
#       without a specified descriptor are encoded with the binding identity, so molecules without stereo
#       annotations keep exactly their fingerprint. False (the default) ignores stereochemistry, as in all HDF
#       results of the paper. Embedding caches of stereo runs carry a "__stereo" suffix.
STEREO: bool = False
# :param QUANTIZE_BITS:
#       If set, every component of the HDF vectors is quantized to 2**QUANTIZE_BITS levels before the models are
#       trained: per embedding dimension, equal-frequency bins are fitted on the training split and each value is
#       replaced by the mean of the training values in its bin (see ``quantize_equal_frequency``). None keeps the
#       float32 vectors. Used for the bit-depth ablation (ex_20, reviewer comment R2.2). The embedding cache
#       always stores the unquantized vectors.
QUANTIZE_BITS: Optional[int] = None
# :param SPECTRUM:
#       Fourier magnitudes of the random codebook vectors (the element vectors and the base vectors of the
#       fractional power encoders), see ``unit_modulus`` of AtomEncoder and ContinuousEncoder. "unit" (default
#       since 2026-10-09, ex_22): unit Fourier magnitudes with random phases (FHRR phasors). "gaussian": the random
#       magnitudes of the original encoder, with which binding concentrates the embeddings on a few Fourier
#       components. Only used in ENCODING_MODE "continuous". Embedding caches of "unit" runs carry a "__unitspec"
#       suffix, so caches of the original encoder are never reused for it.
SPECTRUM: str = 'unit'
# :param GRAPH_ATTRIBUTES:
#       Whether the encodings of the graph size and graph diameter are added to the embedding (default True, the
#       original encoder). False keeps only the message passing part, normalized to unit length as in the original
#       formula normalize(normalize(S) + normalize(G)) without G (ablation ex_22). Without this normalization the
#       norm of the sum-pooled readout would still carry the molecule size. Embedding caches of False runs carry a
#       "__noglobal" suffix.
GRAPH_ATTRIBUTES: bool = True
# :param CODEBOOK_SEED:
#       Seed of the random codebooks (element vectors and fractional power encoders). None (default): SEED, i.e. new
#       codebooks in every repetition. An integer fixes one codebook for all repetitions, as the SI describes HDF (the
#       dictionaries are drawn once), so that the encoding of a full dataset is computed once and shared by all split
#       seeds (cache key "codebook_<seed>" instead of "seed_<SEED>"; ex_27).
CODEBOOK_SEED: Optional[int] = None
# :param HDC_CACHE:
#       Whether the HDF encoding of the dataset is read from and written to the experiment cache (default True). False
#       encodes the dataset in the run without touching the cache, for sweeps in which an encoding serves only one or
#       a few runs (ex_29: new codebooks in every repetition), so that they do not fill the disk with cache files.
HDC_CACHE: bool = True
# :param HDC_ENCODE_BATCH:
#       Number of molecules per chunk of the HDF encoding (default 600, the batch size used so far). The encoder is
#       called chunk by chunk and the embeddings are copied out of each chunk, so that the batch tensors of a chunk
#       are released before the next one (the embeddings returned by forward_graphs are views that keep them alive).
#       Smaller chunks lower the peak memory at large dimensions; the encodings do not depend on the chunk size
#       (ex_29 uses max(16, 600 * 2048 // EMBEDDING_SIZE)).
HDC_ENCODE_BATCH: int = 600
# :param GRAPH_CLOGP:
#       Whether Crippen's logP of each molecule (RDKit MolLogP) is encoded as a further global attribute, next to the
#       graph size and diameter (ex_26). It uses the same fractional power encoding, with a bandwidth of one seventh of
#       the logP range of the dataset between its 1st and 99th percentile (at least 1, see
#       ``make_value_graph_encoder``), and its hypervector is summed with those of the size and diameter. Requires
#       ENCODING_MODE "continuous", GRAPH_ATTRIBUTES and SPECTRUM "unit". Embedding caches of True runs carry a
#       "__clogp" suffix.
GRAPH_CLOGP: bool = False
# :param SPECTRUM_DIAGNOSTIC:
#       If True, the median number of effective Fourier components of the embeddings (participation ratio of the
#       power spectrum, over at most 1000 molecules) is stored as "embedding/effective_components" (ex_22).
SPECTRUM_DIAGNOSTIC: bool = False

# == VISUALIZATION PARAMETERS ==

# :param PLOT_UMAP:
#       A boolean flag that determines whether to plot the UMAP dimensionality reduction of the HDC vectors
#       for the molecular graphs in the dataset.
PLOT_UMAP: bool = False

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(
    e: Experiment,
    index_data_map: dict
) -> None:
    """
    This hook is supposed to implement the processing of the dataset into a suitable vector representation
    that can then be used to train the prediction models on. It gives the dataset in the format of the 
    `index_data_map: Dict[int, dict]` and expects that each dict element in that dataset is updated with 
    a "graph_features" entry that contains the vector representation of the graph.
    """
    
    # --- dataset statistics ---
    # In the first step we need to determine the datasets statistics. Primarily we need the information
    # about the maximum graph size and the maximum graph diameter to be known for the dataset because
    # the encoding of the continuous graph features needs to know these values.
    # Since this is information for which we need to loop through the entire dataset but does not change
    # for each dataset we obviously want to cache that.
    # NOTE: The statistics are computed over the *already-subsampled* index_data_map, so they depend on the
    # molecule set. When subsampling (NUM_DATA < 1.0) we therefore key the cache by NUM_DATA and SEED to
    # avoid a subset's (potentially smaller) max size/diameter being reused for a full-data run — which would
    # miscalibrate the fractional-power encoders. Full-data runs keep the original, seed-independent key.
    if e.NUM_DATA is None or e.NUM_DATA == 1.0:
        stats_cache_name = f'stats_{e.DATASET_NAME}'
    else:
        stats_cache_name = f'stats_{e.DATASET_NAME}__numdata_{e.NUM_DATA}__seed_{e.SEED}'
    @experiment.cache.cached(name=stats_cache_name)
    def dataset_statistics() -> dict:
        
        sizes: List[int] = []
        diameters: List[int] = []
        for index, graph in index_data_map.items():
            smiles: str = graph['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            adj = rdmolops.GetAdjacencyMatrix(mol)
            graph_nx = nx.from_numpy_array(adj)
            
            sizes.append(len(graph_nx.nodes))
            diameters.append(nx.diameter(graph_nx))
            
        return {
            'size': {
                'min': min(sizes),
                'max': max(sizes),
                'mean': sum(sizes) / len(sizes),
                'median': sorted(sizes)[len(sizes) // 2],
            },
            'diameter': {
                'min': min(diameters),
                'max': max(diameters),
                'mean': sum(diameters) / len(diameters),
                'median': sorted(diameters)[len(diameters) // 2],
            }
        }
        
    stats: dict = dataset_statistics()
    pprint(stats)
    
    # --- Constructing HyperNet encoder ---
    # We want to differentiate two modes here: The categorical only mode which is the one that is required 
    # for the decoding and the continuous mode which is the one that performs better for the regression 
    # tasks.
    
    # the codebooks depend on CODEBOOK_SEED if given, otherwise on the split seed as before
    codebook_seed = e.SEED if e.CODEBOOK_SEED is None else e.CODEBOOK_SEED

    if e.SPECTRUM not in ('gaussian', 'unit'):
        raise ValueError(f'SPECTRUM must be "gaussian" or "unit", got {e.SPECTRUM!r}')

    # updated mode with better regression performance.
    if e.ENCODING_MODE == 'continuous':

        node_encoder_map = make_molecule_node_encoder_map_cont(
            dim=e.EMBEDDING_SIZE,
            seed=codebook_seed,
            unit_modulus=(e.SPECTRUM == 'unit'),
            stereo=e.STEREO,
        )
        graph_encoder_map = make_molecule_graph_encoder_map_cont(
            dim=e.EMBEDDING_SIZE,
            seed=codebook_seed,
            max_graph_size=stats['size']['max'],
            max_graph_diameter=stats['diameter']['max'],
            unit_modulus=(e.SPECTRUM == 'unit'),
        )

    # The previous mode.
    elif e.ENCODING_MODE == 'categorical':
        
        node_encoder_map = make_molecule_node_encoder_map(
            dim=e.EMBEDDING_SIZE,
            seed=codebook_seed,
            stereo=e.STEREO,
        )
        graph_encoder_map = {}

    # --- ablation of ex_22 (the default keeps the size and diameter encodings) ---
    if not e.GRAPH_ATTRIBUTES:
        graph_encoder_map = {}

    # --- Crippen's logP as a further global attribute (ex_26) ---
    # The encoder needs the logP range of the whole dataset (like the size and diameter encoders need their maxima),
    # so the values are computed here, before the cached encoding, which then only attaches them to the graph dicts.
    clogp_of: Dict[int, float] = {}
    if e.GRAPH_CLOGP:
        # the gaussian spectrum raises random magnitudes to the power logP / bandwidth (unbounded for negative logP)
        if e.ENCODING_MODE != 'continuous' or not e.GRAPH_ATTRIBUTES or e.SPECTRUM != 'unit':
            raise ValueError('GRAPH_CLOGP requires ENCODING_MODE "continuous", GRAPH_ATTRIBUTES = True and '
                             'SPECTRUM "unit"')
        clogp_of = {index: float(Crippen.MolLogP(Chem.MolFromSmiles(data['graph_repr'])))
                    for index, data in index_data_map.items()}
        # seed offset 50: not used by any node or graph encoder of the maps above (offsets 0-30; the stereo encoder of
        # the branch revision/stereo uses 40)
        graph_encoder_map['graph_clogp'] = make_value_graph_encoder(
            list(clogp_of.values()),
            dim=e.EMBEDDING_SIZE,
            seed=codebook_seed + 50,
            unit_modulus=(e.SPECTRUM == 'unit'),
        )
        e['clogp/min'] = min(clogp_of.values())
        e['clogp/max'] = max(clogp_of.values())
        e['clogp/bandwidth'] = graph_encoder_map['graph_clogp'].bandwidth
        e.log(f' * GRAPH_CLOGP: logP from {e["clogp/min"]:.2f} to {e["clogp/max"]:.2f}, 1st-99th percentile range '
              f'{graph_encoder_map["graph_clogp"].size:.2f}, bandwidth {e["clogp/bandwidth"]:.2f}')


    e.log('creating HyperNet encoder...')
    e.log(f' * DEVICE: {e.DEVICE}')
    e.log(f' * EMBEDDING_SIZE: {e.EMBEDDING_SIZE}')
    e.log(f' * NUM_LAYERS: {e.NUM_LAYERS}')
    e.log(f' * ENCODING_MODE: {e.ENCODING_MODE}')
    e.log(f' * SPECTRUM: {e.SPECTRUM}')
    e.log(f' * GRAPH_ATTRIBUTES: {e.GRAPH_ATTRIBUTES} ({", ".join(graph_encoder_map) or "none"})')
    e.log(f' * STEREO: {e.STEREO}')

    hyper_net = HyperNet(
        hidden_dim=e.EMBEDDING_SIZE,
        depth=e.NUM_LAYERS,
        device=e.DEVICE,
        node_encoder_map=node_encoder_map,
        graph_encoder_map=graph_encoder_map,
        seed=codebook_seed,
        normalize_all=True,
        bidirectional=e.BIDIRECTIONAL,
    )
    
    e.log('saving HyperNet encoder to disk...')
    model_path = os.path.join(e.path, 'hyper_net.pth')
    hyper_net.save_to_path(model_path)

    # --- processing dataset ---
    # After having constructed the HyperNet encoder, we can now use it to process the entire dataset
    # and generate the HDC vectors for each of the molecular graphs in the dataset.
    # The embeddings already depend on SEED (through the random hypervector dictionary) and on
    # EMBEDDING_SIZE / NUM_LAYERS. When subsampling they additionally depend on which molecules were drawn,
    # so we add NUM_DATA to the key to prevent a subsampled run from colliding with a full-data run. Full-data
    # runs keep the original key so existing embedding caches remain valid.
    if e.NUM_DATA is None or e.NUM_DATA == 1.0:
        # the full-data encoding does not depend on the split seed, only on the codebooks
        seed_key = f'seed_{e.SEED}' if e.CODEBOOK_SEED is None else f'codebook_{e.CODEBOOK_SEED}'
        hdc_cache_name = f'hdc_{e.DATASET_NAME}__{seed_key}__size_{e.EMBEDDING_SIZE}__depth_{e.NUM_LAYERS}'
    else:
        # a subsample depends on the split seed (which molecules are drawn) and on the codebooks
        seed_key = f'seed_{e.SEED}' + ('' if e.CODEBOOK_SEED is None else f'__codebook_{e.CODEBOOK_SEED}')
        hdc_cache_name = f'hdc_{e.DATASET_NAME}__numdata_{e.NUM_DATA}__{seed_key}__size_{e.EMBEDDING_SIZE}__depth_{e.NUM_LAYERS}'
    if e.BIDIRECTIONAL:
        hdc_cache_name += '__bidir'
    if e.HYDROGEN_COUNT == 'total':
        hdc_cache_name += '__totalh'
    if e.ENCODING_MODE == 'continuous' and e.SPECTRUM == 'unit':
        hdc_cache_name += '__unitspec'
    if not e.GRAPH_ATTRIBUTES:
        hdc_cache_name += '__noglobal'
    if e.GRAPH_CLOGP:
        hdc_cache_name += '__clogp'
    if e.STEREO:
        hdc_cache_name += '__stereo'
    def process_dataset():
        
        # --- graph representations ---
        # Before we can do the actual forward pass of the encoder, we need to convert the individual
        # elements of the dataset into the graph representations. Only on those graph 
        # representations, the encoder can operate.
        e.log('processing molecules into graphs...')
        time_start_process = time.time()
        # Time of the actual encoding (graph conversion + HyperNet forward pass), excluding dataset
        # statistics, cache (de)serialization and model saving. Only set on a cache miss.
        time_start_encode = time.time()
        graphs: List[dict] = []
        for c, (index, data) in enumerate(index_data_map.items()):
            
            smiles: str = data['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            
            graph = graph_dict_from_mol(mol, hydrogens=e.HYDROGEN_COUNT, stereo=e.STEREO)
            if e.GRAPH_CLOGP:
                graph['graph_clogp'] = clogp_of[index]

            del graph['graph_labels']
            index_data_map[index].update(graph)
            
            graphs.append(graph)
            
            if c % 1000 == 0:
                e.log(f' * {c} molecules done')
                
        time_end_process = time.time()
        e.log(f'processed the dataset after {time_end_process - time_start_process:.2f} seconds')
            
        # --- forward pass ---
        # Finally, the forward pass of the HyperNet encoder can be done on the graph representations
        # to generate the HDC vectors for each of the molecular graphs in the dataset. We then save 
        # that vector representation as the additonal "graph_features" entry in the graph dict, as 
        # this is expected as the outcome of this hook implementation.
        e.log('doing the model forward pass on all the graphs...')
        
        time_start_forward = time.time()
        items = list(index_data_map.items())
        chunk = e.HDC_ENCODE_BATCH
        for start in range(0, len(graphs), chunk):
            results = hyper_net.forward_graphs(graphs[start:start + chunk], batch_size=chunk)
            assert len(results) == len(graphs[start:start + chunk]), 'one encoding per molecule of the chunk'
            for (index, graph), result in zip(items[start:start + chunk], results):
                # a copy: the result is a view that would keep the batch tensors of the chunk alive
                index_data_map[index]['graph_features'] = np.array(result['graph_embedding'])
                # Without graph attributes HyperNet returns the raw sum-pooled readout, whose norm grows with the
                # molecule size. Normalizing it gives the original formula with the size/diameter part removed.
                if not e.GRAPH_ATTRIBUTES:
                    embedding = index_data_map[index]['graph_features']
                    index_data_map[index]['graph_features'] = embedding / np.linalg.norm(embedding)
            del results

        time_end_forward = time.time()
        e.log(f'done the model forward pass after {time_end_forward - time_start_forward:.2f} seconds')
        e['encode_time'] = time_end_forward - time_start_encode
            
        return index_data_map

    if e.HDC_CACHE:
        process_dataset = experiment.cache.cached(name=hdc_cache_name)(process_dataset)
    index_data_map_processed = process_dataset()
    for index in index_data_map:
        index_data_map[index]['graph_features'] = index_data_map_processed[index]['graph_features']

    # how many Fourier components effectively carry the energy of the embeddings (diagnostic of ex_22)
    if e.SPECTRUM_DIAGNOSTIC:
        sample = [np.asarray(index_data_map[index]['graph_features'], dtype=float)
                  for index in list(index_data_map)[:1000]]
        e['embedding/effective_components'] = effective_components(np.stack(sample))
        e['embedding/mean_norm'] = float(np.mean([np.linalg.norm(x) for x in sample]))
        e.log(f' * effective Fourier components of the embeddings: {e["embedding/effective_components"]:.0f} '
              f'of {e.EMBEDDING_SIZE} (median over {len(sample)} molecules), mean norm '
              f'{e["embedding/mean_norm"]:.4f}')


@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict,
                  **kwargs
                  ) -> None:

    # --- quantization ---
    # The bins are fitted on the training split only, so no information of the test molecules leaks in.
    if e.QUANTIZE_BITS is not None:
        e.log(f'quantizing the HDF vectors to {e.QUANTIZE_BITS} bits per dimension...')
        indices = list(index_data_map.keys())
        row_of = {index: row for row, index in enumerate(indices)}
        features = np.array([index_data_map[index]['graph_features'] for index in indices])
        train_rows = np.array([row_of[index] for index in kwargs['train_indices']])
        quantized = quantize_equal_frequency(features, train_rows, e.QUANTIZE_BITS)
        for row, index in enumerate(indices):
            index_data_map[index]['graph_features'] = quantized[row]
        e['quantize/mean_abs_error'] = float(np.abs(quantized - features).mean())
        e['quantize/mean_levels'] = float(np.mean([len(np.unique(quantized[train_rows, d]))
                                                   for d in range(quantized.shape[1])]))
        e.log(f' * mean abs. error {e["quantize/mean_abs_error"]:.2e}, '
              f'{e["quantize/mean_levels"]:.1f} levels per dimension (training split)')
    
    if e.PLOT_UMAP:
        
        e.log('plotting UMAP dimensionality reduction...')
        
        # First of all we need to collect all the HDC vectors for the various graphs in the dataset
        hvs = [data['graph_features'] for data in index_data_map.values()]
        
        reducer = umap.UMAP(
            n_components=2, 
            random_state=e.SEED,
            metric='cosine',
            min_dist=0.0,
            n_neighbors=100,
        )
        reduced = reducer.fit_transform(hvs)
        
        # Extract the class labels from the graph dicts
        if e.DATASET_TYPE == 'regression':
            labels = [data['graph_labels'][0] for data in index_data_map.values()]
            
        if e.DATASET_TYPE == 'classification':
            labels = [np.argmax(data['graph_labels']) for data in index_data_map.values()]
                    
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        ax.set_title('UMAP reduction of HDC vectors\n'
                     '')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # Calculate the 0.05 and 0.95 percentiles
        vmin, vmax = np.percentile(labels, [2, 98])
        
        # Clip the labels to the 0.05 and 0.95 percentiles
        clipped_labels = np.clip(labels, vmin, vmax)
        
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1], 
            c=clipped_labels, 
            marker='.',
            cmap='bwr', 
            alpha=0.5,
            edgecolors='none',
            s=10  # Adjust the size of the scatter points
        )
        
        # # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('target')
        
        fig_path = os.path.join(e.path, 'umap_reduction.png')
        fig.savefig(fig_path, dpi=600)
    

experiment.run_if_main()