"""
Randomly initialized, *untrained* GNNs as training-free molecular encoders.

This experiment is the control for the question whether the predictive performance of hyperdimensional
fingerprints (HDF) stems from the HDC algebra itself or simply from the multi-hop message passing
structure. It uses the same GCN / GIN / GATv2 architectures as the trained baseline
(``predict_molecules__gnn.py``), but never trains them: the weights stay at their random initialization
and the sum-pooled graph embedding is used as a fixed vector representation, exactly like HDF. The
simple downstream models of the base experiment (MLP, KNN) are then trained on top of that
representation.

To isolate the encoding mechanism, the defaults match the HDF configuration of the main comparison:
the embedding width equals the HDF dimension (EMBEDDING_SIZE = 2048), the number of message passing
layers equals the HDF depth (NUM_LAYERS = 2), and the node features carry exactly the atom attributes
that HDF encodes (NODE_FEATURES = 'hdf').
"""
import time
from typing import List, Literal

import torch
import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from torch_geometric.loader import DataLoader

from graph_hdc.baselines.gnn import GNN_CLASSES
from graph_hdc.baselines.gnn import hdf_matched_graph
from graph_hdc.baselines.gnn import build_pyg_list


# == ENCODER PARAMETERS ==

# :param GNN_ARCH:
#       The GNN architecture whose random initialization is used as the encoder: 'gcn', 'gin' or 'gatv2'.
GNN_ARCH: Literal['gcn', 'gin', 'gatv2'] = 'gin'
# :param EMBEDDING_SIZE:
#       The width of all message passing layers and therefore the dimension of the pooled graph embedding.
EMBEDDING_SIZE: int = 2048
# :param NUM_LAYERS:
#       The number of message passing layers.
NUM_LAYERS: int = 2
# :param NODE_FEATURES:
#       'hdf' restricts the input node features to the atom attributes that HDF encodes (element, heavy
#       atom degree, implicit H count; no bond types). 'default' uses the full ChemMatData featurization.
NODE_FEATURES: Literal['default', 'hdf'] = 'hdf'
# :param ENCODE_BATCH_SIZE:
#       The number of graphs per batch during the encoding forward pass. Kept small because a width-2048
#       GATv2 materializes several (num_edges x 5 * 2048) tensors: a batch of 512 COMPAS graphs needs ~16 GB,
#       while three runs share one 24 GB GPU on the cluster.
ENCODE_BATCH_SIZE: int = 64
# :param DEVICE:
#       The device for the encoding forward pass ('cpu' or 'cuda'). Encoding on the GPU and training the
#       Lightning MLP afterwards in the same process was verified to work (ex_14 uses 'cuda'; ~18x faster).
DEVICE: str = 'cpu'

# == MODEL PARAMETERS ==

# :param MODELS:
#       The downstream models trained on top of the frozen random GNN embeddings.
MODELS: List[str] = [
    'neural_net2',
    'k_neighbors',
]

# == EXPERIMENT PARAMETERS ==

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
    """
    Encode every graph with a frozen, randomly initialized GNN and store the sum-pooled embedding as
    the "graph_features" vector representation.

    The initialization is seeded with SEED, so every repetition of the experiment draws a new random
    encoder (just like every HDF repetition draws a new random hypervector dictionary) and the variance
    across repetitions includes the variance due to the random initialization.
    """
    # The encoding time covers the featurization of the molecules (as HDF's encode_time covers its
    # graph conversion) plus the forward pass.
    time_start = time.time()
    if e.NODE_FEATURES == 'hdf':
        for data in index_data_map.values():
            hdf_matched_graph(data)

    indices = list(index_data_map.keys())
    example_graph = index_data_map[indices[0]]

    # Default PyTorch / PyG initialization of the same model class as the trained baseline. The
    # prediction head is not used; ``embed`` stops after the sum pooling.
    torch.manual_seed(e.SEED)
    model = GNN_CLASSES[e.GNN_ARCH](
        input_dim=example_graph['node_attributes'].shape[1],
        output_dim=1,
        output_type='regression',
        conv_units=[e.EMBEDDING_SIZE] * e.NUM_LAYERS,
        dense_units=[],
    )
    model.eval()
    model.to(e.DEVICE)
    e['encoder/num_params'] = sum(p.numel() for p in model.parameters())
    e.log(f'created random {e.GNN_ARCH} encoder with {e["encoder/num_params"]} parameters')

    data_loader = DataLoader(
        build_pyg_list(index_data_map, indices),
        batch_size=e.ENCODE_BATCH_SIZE,
        shuffle=False,
    )
    embeddings = []
    with torch.no_grad():
        for data in data_loader:
            embeddings.append(model.embed(data.to(e.DEVICE)).cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    e['encode_time'] = time.time() - time_start
    e.log(f'encoded {len(indices)} graphs after {e["encode_time"]:.2f} seconds')

    # Release the encoder and its cached GPU memory before the downstream models are trained, so that
    # the other runs sharing the GPU are not squeezed.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for index, embedding in zip(indices, embeddings):
        index_data_map[index]['graph_features'] = embedding


experiment.run_if_main()
