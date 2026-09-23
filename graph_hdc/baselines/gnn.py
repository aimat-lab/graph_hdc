"""
Graph neural network baselines for the fingerprint experiments.

This module holds the PyTorch Geometric GNN models (GCN, GIN, GATv2) and the featurization helpers which
are shared by two experiment families:

- ``predict_molecules__gnn.py`` trains the GNNs end-to-end on the prediction target.
- ``predict_molecules__gnn_random.py`` uses the *same* architectures with frozen random weights as a
  training-free graph encoder whose pooled embedding is then fed to the simple downstream models (MLP, KNN).

Keeping both on one implementation guarantees that the only difference between the "trained" and the
"random-init" baselines is whether the weights were optimized.
"""
import time
import copy
import inspect
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from rdkit import Chem
from torchmetrics import MeanAbsoluteError
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import Data

from graph_hdc.special.molecules import make_molecule_node_encoder_map_cont


# == HDF-MATCHED FEATURIZATION ==

# The atomic numbers known to the continuous HDF atom encoder. We read them from the default argument of
# the encoder factory instead of copying the list so that the GNN inputs can never drift out of sync
# with what HDF actually encodes.
HDF_ATOMS: List[int] = [
    int(z) for z in inspect.signature(make_molecule_node_encoder_map_cont).parameters['atoms'].default
]
# HDF encodes the heavy-atom degree and the number of implicit hydrogens with fractional power encoders
# of size 10, so we one-hot the same range (values above are clipped into the last bucket).
HDF_MAX_DEGREE: int = 10
HDF_MAX_HYDROGENS: int = 10
# one-hot atoms (+1 "other" bucket), one-hot degree, one-hot implicit hydrogens
HDF_NODE_DIM: int = len(HDF_ATOMS) + 1 + HDF_MAX_DEGREE + HDF_MAX_HYDROGENS


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=float)
    vec[min(index, size - 1)] = 1.0
    return vec


def hdf_matched_graph(graph: dict) -> dict:
    """
    Replace the node and edge features of the given ``graph`` dict with the information that is
    available to the hyperdimensional fingerprint (HDF) encoder, and nothing more.

    The default ChemMatData featurization gives a GNN considerably more information than HDF receives
    (hybridization, aromaticity, ring membership, charge, mass, Crippen contributions). To make the
    GNN comparison a test of the *encoding mechanism* rather than of the input features, this function
    rebuilds the graph from its SMILES with exactly the HDF atom attributes:

    - one-hot atomic number over the HDF atom list (plus one "other" bucket)
    - one-hot heavy-atom degree (the HDF ``node_degrees``)
    - one-hot number of implicit hydrogens (the HDF ``node_valences``)

    HDF does not encode bond types, so the edges only carry a constant dummy attribute. Edges are
    stored in both directions as required for PyG message passing.

    The dict is modified in place and also returned. Graph labels and all other entries are kept.

    .. code-block:: python

        graph = hdf_matched_graph({'graph_repr': 'CCO', 'graph_labels': np.array([1.0])})
        graph['node_attributes'].shape  # (3, HDF_NODE_DIM)

    :param graph: A graph dict which contains at least the "graph_repr" SMILES string.

    :returns: The same graph dict with updated node_attributes, edge_indices and edge_attributes.
    """
    mol = Chem.MolFromSmiles(graph['graph_repr'])

    atom_index = {z: i for i, z in enumerate(HDF_ATOMS)}
    node_attributes = []
    for atom in mol.GetAtoms():
        node_attributes.append(np.concatenate([
            _one_hot(atom_index.get(atom.GetAtomicNum(), len(HDF_ATOMS)), len(HDF_ATOMS) + 1),
            _one_hot(atom.GetDegree(), HDF_MAX_DEGREE),
            _one_hot(atom.GetNumImplicitHs(), HDF_MAX_HYDROGENS),
        ]))

    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [(i, j), (j, i)]

    graph['node_indices'] = np.arange(mol.GetNumAtoms())
    graph['node_attributes'] = np.array(node_attributes, dtype=float)
    graph['edge_indices'] = np.array(edge_indices, dtype=int).reshape(-1, 2)
    graph['edge_attributes'] = np.ones((len(edge_indices), 1), dtype=float)
    return graph


def pyg_data_from_graph(graph: dict) -> Data:
    """
    Convert a single graph dict into a PyG ``Data`` object with node features ``x``, ``edge_index``,
    ``edge_attr`` and (if present) the target ``y``.
    """
    data = Data(
        x=torch.tensor(graph['node_attributes'], dtype=torch.float),
        edge_attr=torch.tensor(graph['edge_attributes'], dtype=torch.float),
        edge_index=torch.tensor(np.asarray(graph['edge_indices']).T, dtype=torch.long),
    )
    if 'graph_labels' in graph:
        data.y = torch.tensor(graph['graph_labels'], dtype=torch.float)
    return data


def build_pyg_list(index_data_map: dict, indices: List[int]) -> List[Data]:
    """
    Pre-compute the PyG ``Data`` objects for the given ``indices`` once.

    Converting every graph on-the-fly inside the data loader (as the older ``LazyGraphDataset`` did)
    repeats the same conversion in every epoch, which dominates the runtime on the large datasets
    (QM9, COMPAS). Molecular graphs are small, so holding all of them in memory is cheap.
    """
    return [pyg_data_from_graph(index_data_map[index]) for index in indices]


# == TRAINING CALLBACKS ==

class BestModelRestorer(pl.Callback):
    """
    This class implements a PyTorch Lightning callback which will restore the model weights to
    that state which achieved the best validation loss observed during the training process.

    This is done by monitoring a specific metric (e.g. 'val_loss') and saving the model state
    whenever the monitored metric improves. Using a hook at the very end of the training, the
    model weights are reset to that best state. The epoch and wall time of the best state are
    recorded so that the experiment can report when the model actually converged.
    """

    def __init__(self,
                 monitor: str = "val_loss",
                 mode: str = "min"
                 ) -> None:
        super().__init__()
        self.monitor = monitor
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'.")
        self.mode = mode

        # This will variable will store the best score observed during the training.
        self.best_score: float = None
        # This will store the best model state dict associated with the best score.
        self.best_state_dict = None
        # This will store the time and epoch when the best score was achieved.
        self.best_time = None
        self.best_epoch = None
        # Per-epoch learning curve: list of (epoch, monitored validation metric).
        self.history = []

    def on_fit_start(self, trainer, pl_module):
        """
        Initialize the best score before starting the fit.
        """
        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = -float("inf")
        self.best_state_dict = None

    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of the validation loop. We check whether the monitored metric improved and
        if so, store the model state dict.
        """
        # The sanity check runs validation before any training step; that state is not a candidate.
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)

        if current_score is None:
            # Metric not found, cannot update best score
            return
        self.history.append((trainer.current_epoch, float(current_score)))

        if (
            (self.mode == "min" and current_score < self.best_score) or
            (self.mode == "max" and current_score > self.best_score)
        ):
            # Update best score and store model weights
            self.best_score = current_score
            self.best_state_dict = {
                k: copy.deepcopy(v.detach().cpu().clone())
                for k, v in pl_module.state_dict().items()
            }
            self.best_time = time.time()
            self.best_epoch = trainer.current_epoch

    def on_train_end(self, trainer, pl_module):
        """
        At the end of training, restore the model to the best recorded state.
        """
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)
            trainer.print(
                f"Restored the best model with {self.monitor}={self.best_score:.4f} "
                f"from epoch {self.best_epoch}."
            )


# == GNN MODELS ==

class GnnModel(pl.LightningModule):
    """
    Base class for the message passing GNNs: a linear node embedding, a stack of graph convolution
    layers with LeakyReLU activations, sum pooling and a dense prediction head.

    Subclasses only define the type of convolution through :meth:`make_conv`. The graph-level
    embedding before the prediction head is available through :meth:`embed`, which is what the
    random-init baseline uses as a frozen, training-free encoder.

    :param early_stopping_patience: If not None, training stops once the validation metric has not
        improved for that many epochs. The best weights are always restored at the end of training.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 output_type: Literal['classification', 'regression'],
                 conv_units: List[int] = [64, 64, 64],
                 dense_units: List[int] = [64, 32],
                 learning_rate: float = 1e-3,
                 early_stopping_patience: Optional[int] = None,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_type = output_type
        self.output_dim = output_dim
        self.conv_units = conv_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        self.lay_act = nn.LeakyReLU()

        # Define loss function based on output type
        if output_type == 'classification':
            self.loss = nn.CrossEntropyLoss()
            self.metric = nn.CrossEntropyLoss()
        elif output_type == 'regression':
            self.loss = nn.MSELoss()
            self.metric = MeanAbsoluteError(num_outputs=output_dim)

        self.lay_embedd = nn.Linear(input_dim, conv_units[0])

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        prev_units = conv_units[0]
        for units in conv_units:
            self.conv_layers.append(self.make_conv(prev_units, units))
            prev_units = units

        # Pooling layer
        self.lay_pool = SumAggregation()

        # Create dense layers
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            lay = nn.Sequential(
                nn.Linear(prev_units, units),
                nn.BatchNorm1d(units),
            )
            self.dense_layers.append(lay)
            prev_units = units

        # Final output layer
        lay_final = nn.Linear(prev_units, output_dim)
        self.dense_layers.append(lay_final)

    def make_conv(self, in_units: int, out_units: int) -> nn.Module:
        raise NotImplementedError()

    def embed(self, data) -> torch.Tensor:
        """
        Graph-level embedding: node embedding, message passing and sum pooling, without the head.

        :param data: A batch of graph data.
        :return: Tensor of shape (num_graphs, conv_units[-1]).
        """
        x, edge_index = data.x, data.edge_index
        node_emb = self.lay_embedd(x)
        for conv in self.conv_layers:
            node_emb = conv(node_emb, edge_index)
            node_emb = self.lay_act(node_emb)

        # Pooling node embeddings to get graph-level embedding
        return self.lay_pool(node_emb, data.batch)

    def forward(self, data):
        """
        Forward pass through the model.

        :param data: A batch of graph data.
        :return: The output predictions of the model.
        """
        out = self.embed(data)

        # Pass through dense layers
        for dense in self.dense_layers[:-1]:
            out = dense(out)
            out = self.lay_act(out)

        # Final output layer
        out = self.dense_layers[-1](out)

        return out

    def training_step(self, data, batch_idx):
        output = self(data)
        loss = self.loss(output, data.y.view(output.shape))
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        output = self(data)
        target = data.y.view(output.shape)

        loss = self.loss(output, target)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=data.num_graphs)

        if self.output_type == 'regression':
            metric = self.metric(output, target)
        elif self.output_type == 'classification':
            output = torch.softmax(output, dim=1)
            labels = torch.argmax(target, dim=1)
            metric = self.metric(output, labels)

        self.log('val_metric', metric, prog_bar=True, on_epoch=True, batch_size=data.num_graphs)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_callbacks(self):
        """
        Best-checkpoint selection on the validation metric, plus optional early stopping on the
        same metric.
        """
        self.model_restorer = BestModelRestorer(
            monitor='val_metric',
            mode='min'
        )
        callbacks = [self.model_restorer]
        if self.early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                monitor='val_metric',
                mode='min',
                patience=self.early_stopping_patience,
            )
            callbacks.append(self.early_stopping)
        return callbacks


class GcnModel(GnnModel):
    """
    A Graph Convolutional Network (GCN) implemented using PyTorch Lightning.
    """

    def make_conv(self, in_units: int, out_units: int) -> nn.Module:
        return GCNConv(
            in_channels=in_units,
            out_channels=out_units,
            improved=True,
            add_self_loops=True,
        )


class GinModel(GnnModel):
    """
    A Graph Isomorphism Network (GIN) implemented using PyTorch Lightning.
    """

    def make_conv(self, in_units: int, out_units: int) -> nn.Module:
        return GINConv(
            nn.Sequential(
                nn.Linear(in_units, 2 * out_units),
                nn.BatchNorm1d(2 * out_units),
                nn.LeakyReLU(),
                nn.Linear(2 * out_units, out_units),
            ),
            train_eps=True,
        )


class Gatv2Model(GnnModel):
    """
    A Graph Attention Network v2 (GATv2) implemented using PyTorch Lightning.
    """

    def make_conv(self, in_units: int, out_units: int) -> nn.Module:
        return GATv2Conv(
            in_channels=in_units,
            out_channels=out_units,
            heads=5,
            concat=False,
            dropout=0.0,
            add_self_loops=True,
        )


GNN_CLASSES = {
    'gcn': GcnModel,
    'gin': GinModel,
    'gatv2': Gatv2Model,
}
