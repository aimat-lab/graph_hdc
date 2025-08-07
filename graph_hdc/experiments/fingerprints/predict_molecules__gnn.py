import os
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Any, Literal

import umap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from scipy.special import softmax
from rdkit.Chem.Crippen import MolLogP
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data.processing import MoleculeProcessing, OneHotEncoder
from rdkit import Chem
from torchmetrics import R2Score
from torchmetrics import Accuracy, F1Score
from torch_geometric.nn import GCNConv, GCN2Conv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# from visual_graph_datasets.data import nx_from_graph
from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.special.molecules import make_molecule_node_encoder_map
from chem_mat_data.main import pyg_data_list_from_graphs

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'bace'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'classification'
# :param NUM_VAL:
#       The number of validation samples to be used for the evaluation of the models during training.
NUM_VAL: int = 0.1

# == MODEL PARAMETERS ==

# :param MODELS:
#       A list of strings specifying the types of GNN models to be used in the experiment.
#       Possible values are 'gin', 'gatv2', and 'gcn'.
MODELS: List[str] = [
    'gcn',
    'gin',
    'gatv2',
]

# :param CONV_UNITS:
#       A list of integers specifying the number of units in each convolutional layer of the GNN models.
CONV_UNITS: List[int] = [256, 256, 256]
# :param DENSE_UNITS:
#       A list of integers specifying the number of units in each dense (fully connected) layer of the GNN models.
DENSE_UNITS: List[int] = [128, 64]
# :param BATCH_SIZE:
#       The size of the batches to be used during training. This parameter determines the number of samples
#       that are processed in parallel during the training of the model.
BATCH_SIZE: int = 32
# :param EPOCHS:
#       The number of epochs to be used for the training of the model. This parameter determines the number of
#       times the model will be trained on the entire dataset.
EPOCHS: int = 100
# :param LEARNING_RATE:
#       The learning rate to be used for the training of the model. This parameter determines the step size that
#       is used to update the model parameters during training.
LEARNING_RATE: float = 1e-3
# :param DEVICE:
#       The device to be used for the training of the model. This parameter can be set to 'cuda:0' to use the
#       GPU for training, or to 'cpu' to use the CPU.
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
#DEVICE: str = "cpu"

# == VISUALIZATION PARAMETERS ==

# :param PLOT_UMAP:
#       A boolean flag that determines whether to plot the UMAP dimensionality reduction of the HDC vectors
#       for the molecular graphs in the dataset.
PLOT_UMAP: bool = False

# == GNN MODELS ==

class BestModelRestorer(pl.Callback):
    
    def __init__(self, monitor: str = "val_loss", mode: str = "min"):
        """
        Args:
            monitor: Metric name to monitor (e.g. 'val_loss').
            mode: One of {'min', 'max'}. In 'min' mode, training seeks to minimize the monitored quantity,
                  in 'max' mode it seeks to maximize it.
        """
        super().__init__()
        self.monitor = monitor
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'.")
        self.mode = mode

        self.best_score = None
        self.best_state_dict = None
        self.best_time = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize the best score before starting the fit."""
        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = -float("inf")
        self.best_state_dict = None

    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of the validation loop. We check whether the monitored metric improved
        and if so, store the model state dict and log the improvement.
        """
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)

        if current_score is None:
            # Metric not found, cannot update best score
            return

        if (
            (self.mode == "min" and current_score < self.best_score) or
            (self.mode == "max" and current_score > self.best_score)
        ):
            # Update best score and store model weights
            self.best_score = current_score
            self.best_state_dict = {
                k: v.cpu() for k, v in pl_module.state_dict().items()
            }
            self.best_time = time.time()

            # Log the new best score (if the logger is available)
            if trainer.logger is not None:
                trainer.logger.log_metrics({f"best_{self.monitor}": current_score}, step=trainer.global_step)
                
            # You could also print a message if desired:
            trainer.print(
                f"New best {self.monitor}={current_score:.4f} at step={trainer.global_step}."
            )

    def on_train_end(self, trainer, pl_module):
        """At the end of training, restore the model to the best recorded state."""
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)
            trainer.print(
                f"Restored the best model with {self.monitor}={self.best_score:.4f}."
            )


class GnnModel(pl.LightningModule):
    
    def __init__(self,
                 output_type: Literal['classification', 'regression'],
                 output_dim: int,
                 learning_rate: float = 1e-4,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.output_type = output_type
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        self.lay_act = nn.LeakyReLU()
        
        # Define loss function based on output type
        if output_type == 'classification':
            self.loss = nn.CrossEntropyLoss()
        elif output_type == 'regression':
            self.loss = nn.MSELoss()
            
        if self.output_type == 'regression':
            self.metric = R2Score()
        elif self.output_type == 'classification':
            self.metric = Accuracy(
                task='multiclass', 
                num_classes=output_dim, 
                top_k=1
            )
    
    def training_step(self, data, batch_idx):
        """
        Training step for the GIN model.

        :param data: A batch of graph data.
        :param batch_idx: The index of the batch.
        :return: The training loss for the batch.
        """
        # Forward pass
        output = self(data)

        # Compute loss
        loss = self.loss(output, data.y.view(output.shape))

        # Log training loss
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, data, batch_idx):
        """
        Validation step for the GIN model.

        :param data: A batch of graph data.
        :param batch_idx: The index of the batch.
        :return: The validation loss for the batch.
        """
        # Forward pass
        output = self(data)
        target = data.y.view(output.shape)

        # Compute loss
        loss = self.loss(output, data.y.view(output.shape))
        # Log validation loss
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        if self.output_type == 'regression':
            metric = self.metric(output, target)
        elif self.output_type == 'classification':
            output = torch.softmax(output, dim=1)
            labels = torch.argmax(target, dim=1)
            metric = self.metric(output, labels)

        self.log('val_metric', metric, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        :return: The optimizer to be used for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def configure_callbacks(self):
        """
        Configures the callbacks for training.

        :return: A list of callbacks to be used during training.
        """
        early_stopping = pl.callbacks.EarlyStopping(
            monitor='val_metric',
            mode='max',
            patience=10,
            verbose=True
        )
        self.model_restorer = BestModelRestorer(
            monitor='val_metric',
            mode='max'
        )
        return [self.model_restorer]


class GcnModel(GnnModel):
    """
    A Graph Convolutional Network (GCN) implemented using PyTorch Lightning.
    
    This model is designed for both classification and regression tasks on graph-structured data.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 output_type: Literal['classification', 'regression'],
                 conv_units: List[int] = [64, 64, 64],
                 dense_units: List[int] = [64, 32],
                 learning_rate: float = 1e-3,
                 ):
        """
        Initializes the GCN model with the given parameters.

        :param input_dim: The dimensionality of the input node features.
        :param output_dim: The dimensionality of the output predictions.
        :param output_type: The type of the output, either 'classification' or 'regression'.
        :param conv_units: A list of integers specifying the number of units in each convolutional layer.
        :param dense_units: A list of integers specifying the number of units in each dense (fully connected) layer.
        """
        super(GcnModel, self).__init__(
            output_type=output_type,
            output_dim=output_dim,
            learning_rate=learning_rate
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_units = conv_units
        self.dense_units = dense_units

        self.lay_embedd = nn.Linear(input_dim, conv_units[0])

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        prev_units = conv_units[0]
        for units in conv_units:
            lay = GCNConv(
                in_channels=prev_units, 
                out_channels=units,
                improved=True,
                add_self_loops=True,
            )
            self.conv_layers.append(lay)
            prev_units = units

        # Pooling layer
        self.lay_pool = SumAggregation()

        # Create dense layers
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            lay = nn.Linear(prev_units, units)
            self.dense_layers.append(lay)
            prev_units = units

        # Final output layer
        lay_final = nn.Linear(prev_units, output_dim)
        self.dense_layers.append(lay_final)

    def forward(self, data):
        """
        Forward pass through the GCN model.

        :param data: A batch of graph data.
        :return: The output predictions of the model.
        """
        x, edge_index = data.x, data.edge_index
        node_emb = self.lay_embedd(x)
        for conv in self.conv_layers:
            node_emb = conv(node_emb, edge_index)
            node_emb = self.lay_act(node_emb)

        # Pooling node embeddings to get graph-level embedding
        graph_emb = self.lay_pool(node_emb, data.batch)
        out = graph_emb

        # Pass through dense layers
        for dense in self.dense_layers[:-1]:
            out = dense(out)
            out = self.lay_act(out)

        # Final output layer
        out = self.dense_layers[-1](out)

        return out
    
    
class GinModel(GnnModel):
    """
    A Graph Isomorphism Network (GIN) implemented using PyTorch Lightning.
    
    This model is designed for both classification and regression tasks on graph-structured data.
    """

    def __init__(self,
                    input_dim: int,
                    output_dim: int,
                    output_type: Literal['classification', 'regression'],
                    conv_units: List[int] = [64, 64, 64],
                    dense_units: List[int] = [64, 32],
                    learning_rate: float = 1e-3,
                    ):
        """
        Initializes the GIN model with the given parameters.

        :param input_dim: The dimensionality of the input node features.
        :param output_dim: The dimensionality of the output predictions.
        :param output_type: The type of the output, either 'classification' or 'regression'.
        :param conv_units: A list of integers specifying the number of units in each convolutional layer.
        :param dense_units: A list of integers specifying the number of units in each dense (fully connected) layer.
        """
        super(GinModel, self).__init__(
            output_type=output_type,
            output_dim=output_dim,
            learning_rate=learning_rate
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_units = conv_units
        self.dense_units = dense_units

        self.lay_embedd = nn.Linear(input_dim, conv_units[0])

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        prev_units = conv_units[0]
        for units in conv_units:
            lay = GINConv(
                nn.Sequential(
                    nn.Linear(prev_units, 2 * units),
                    nn.BatchNorm1d(2 * units),
                    nn.LeakyReLU(),
                    nn.Linear(2 * units, units),
                    nn.Dropout1d(0.1),
                ),
                train_eps=True,
            )
            self.conv_layers.append(lay)
            prev_units = units

        # Pooling layer
        self.lay_pool = SumAggregation()

        # Create dense layers
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            lay = nn.Linear(prev_units, units)
            self.dense_layers.append(lay)
            prev_units = units

        # Final output layer
        lay_final = nn.Linear(prev_units, output_dim)
        self.dense_layers.append(lay_final)

    def forward(self, data):
        """
        Forward pass through the GIN model.

        :param data: A batch of graph data.
        :return: The output predictions of the model.
        """
        x, edge_index = data.x, data.edge_index
        node_emb = self.lay_embedd(x)
        for conv in self.conv_layers:
            node_emb = conv(node_emb, edge_index)
            node_emb = self.lay_act(node_emb)

        # Pooling node embeddings to get graph-level embedding
        graph_emb = self.lay_pool(node_emb, data.batch)
        out = graph_emb

        # Pass through dense layers
        for dense in self.dense_layers[:-1]:
            out = dense(out)
            out = self.lay_act(out)

        # Final output layer
        out = self.dense_layers[-1](out)

        return out
    
    
class Gatv2Model(GnnModel):
    """
    A Graph Attention Network v2 (GATv2) implemented using PyTorch Lightning.
    
    This model is designed for both classification and regression tasks on graph-structured data.
    """

    def __init__(self,
                    input_dim: int,
                    output_dim: int,
                    output_type: Literal['classification', 'regression'],
                    conv_units: List[int] = [64, 64, 64],
                    dense_units: List[int] = [64, 32],
                    learning_rate: float = 1e-3,
                    ):
        """
        Initializes the GATv2 model with the given parameters.

        :param input_dim: The dimensionality of the input node features.
        :param output_dim: The dimensionality of the output predictions.
        :param output_type: The type of the output, either 'classification' or 'regression'.
        :param conv_units: A list of integers specifying the number of units in each convolutional layer.
        :param dense_units: A list of integers specifying the number of units in each dense (fully connected) layer.
        """
        super(Gatv2Model, self).__init__(
            output_type=output_type,
            output_dim=output_dim,
            learning_rate=learning_rate
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_units = conv_units
        self.dense_units = dense_units

        self.lay_embedd = nn.Linear(input_dim, conv_units[0])

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        prev_units = conv_units[0]
        for units in conv_units:
            lay = GATv2Conv(
                in_channels=prev_units, 
                out_channels=units,
                heads=3,
                concat=False,
                dropout=0.0,
                add_self_loops=True,
            )
            self.conv_layers.append(lay)
            prev_units = units

        # Pooling layer
        self.lay_pool = SumAggregation()

        # Create dense layers
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            lay = nn.Linear(prev_units, units)
            self.dense_layers.append(lay)
            prev_units = units

        # Final output layer
        lay_final = nn.Linear(prev_units, output_dim)
        self.dense_layers.append(lay_final)

    def forward(self, data):
        """
        Forward pass through the GATv2 model.

        :param data: A batch of graph data.
        :return: The output predictions of the model.
        """
        x, edge_index = data.x, data.edge_index
        node_emb = self.lay_embedd(x)
        for conv in self.conv_layers:
            node_emb = conv(node_emb, edge_index)
            node_emb = self.lay_act(node_emb)

        # Pooling node embeddings to get graph-level embedding
        graph_emb = self.lay_pool(node_emb, data.batch)
        out = graph_emb

        # Pass through dense layers
        for dense in self.dense_layers[:-1]:
            out = dense(out)
            out = self.lay_act(out)

        # Final output layer
        out = self.dense_layers[-1](out)

        return out

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('train_model__gcn', replace=False, default=True)
def train_model__gcn(e: Experiment,
                     index_data_map: dict,
                     train_indices: list[int],
                     val_indices: list[int],
                     ) -> Any:
    """
    This hook is called during the training period of the experiment to train a model of the "gcn" type.
    ---
    In this specific implementation, the data is first converted into PyTorch Geometric Data objects and then
    used to train a Graph Convolutional Network (GCN) model for the given number of EPOCHS. The trained model 
    is then returned. 
    """
    # Extract the graphs corresponding to the training indices
    graphs_train = [index_data_map[i] for i in train_indices]
    example_graph = graphs_train[0]
    
    # Extract the graphs corresponding to the validation indices
    graphs_val = [index_data_map[i] for i in val_indices]
    
    # Convert the list of graphs into a list of PyTorch Geometric Data objects
    data_list_train: List[Data] = pyg_data_list_from_graphs(graphs_train)
    data_list_val: List[Data] = pyg_data_list_from_graphs(graphs_val)
    
    # Create a DataLoader to handle batching of the data during training
    data_loader_train = DataLoader(data_list_train, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_val = DataLoader(data_list_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize the GCN model with the appropriate input and output dimensions
    model = GcnModel(
        input_dim=example_graph['node_attributes'].shape[1],
        output_dim=example_graph['graph_labels'].shape[0],
        output_type=e.DATASET_TYPE,
        conv_units=e.CONV_UNITS,
        dense_units=e.DENSE_UNITS,
        learning_rate=e.LEARNING_RATE,
    )
    
    # Use PyTorch Lightning's Trainer to handle the training loop
    time_start = time.time()
    trainer = pl.Trainer(max_epochs=e.EPOCHS, logger=False)
    trainer.fit(model, data_loader_train, data_loader_val)
    
    time_end = model.model_restorer.best_time
    e['train_time/gcn'] = time_end - time_start
        
    model.eval()
        
    # Return the trained model
    return model


@experiment.hook('train_model__gin', replace=False, default=True)
def train_model__gin(e: Experiment,
                        index_data_map: dict,
                        train_indices: list[int],
                        val_indices: list[int],
                        ) -> Any:
    """
    This hook is called during the training period of the experiment to train a model of the "gin" type.
    ---
    In this specific implementation, the data is first converted into PyTorch Geometric Data objects and then
    used to train a Graph Isomorphism Network (GIN) model for the given number of EPOCHS. The trained model 
    is then returned. 
    """
    # Extract the graphs corresponding to the training indices
    graphs_train = [index_data_map[i] for i in train_indices]
    example_graph = graphs_train[0]
    
    # Extract the graphs corresponding to the validation indices
    graphs_val = [index_data_map[i] for i in val_indices]
    
    # Convert the list of graphs into a list of PyTorch Geometric Data objects
    data_list_train: List[Data] = pyg_data_list_from_graphs(graphs_train)
    data_list_val: List[Data] = pyg_data_list_from_graphs(graphs_val)
    
    # Create a DataLoader to handle batching of the data during training
    data_loader_train = DataLoader(data_list_train, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_val = DataLoader(data_list_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize the GIN model with the appropriate input and output dimensions
    model = GinModel(
        input_dim=example_graph['node_attributes'].shape[1],
        output_dim=example_graph['graph_labels'].shape[0],
        output_type=e.DATASET_TYPE,
        conv_units=e.CONV_UNITS,
        dense_units=e.DENSE_UNITS,
        learning_rate=e.LEARNING_RATE,
    )
    
    # Use PyTorch Lightning's Trainer to handle the training loop
    time_start = time.time()
    trainer = pl.Trainer(max_epochs=e.EPOCHS, logger=False)
    trainer.fit(model, data_loader_train, data_loader_val)
        
    time_end = model.model_restorer.best_time
    e['train_time/gin'] = time_end - time_start
        
    model.eval()
        
    # Return the trained model
    return model


@experiment.hook('train_model__gatv2', replace=False, default=True)
def train_model__gatv2(e: Experiment,
                        index_data_map: dict,
                        train_indices: list[int],
                        val_indices: list[int],
                        ) -> Any:
    """
    This hook is called during the training period of the experiment to train a model of the "gatv2" type.
    ---
    In this specific implementation, the data is first converted into PyTorch Geometric Data objects and then
    used to train a Graph Attention Network v2 (GATv2) model for the given number of EPOCHS. The trained model 
    is then returned. 
    """
    # Extract the graphs corresponding to the training indices
    graphs_train = [index_data_map[i] for i in train_indices]
    example_graph = graphs_train[0]
    
    # Extract the graphs corresponding to the validation indices
    graphs_val = [index_data_map[i] for i in val_indices]
    
    # Convert the list of graphs into a list of PyTorch Geometric Data objects
    data_list_train: List[Data] = pyg_data_list_from_graphs(graphs_train)
    data_list_val: List[Data] = pyg_data_list_from_graphs(graphs_val)
    
    # Create a DataLoader to handle batching of the data during training
    data_loader_train = DataLoader(data_list_train, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_val = DataLoader(data_list_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize the GATv2 model with the appropriate input and output dimensions
    model = Gatv2Model(
        input_dim=example_graph['node_attributes'].shape[1],
        output_dim=example_graph['graph_labels'].shape[0],
        output_type=e.DATASET_TYPE,
        conv_units=e.CONV_UNITS,
        dense_units=e.DENSE_UNITS,
        learning_rate=e.LEARNING_RATE,
    )
    
    # Use PyTorch Lightning's Trainer to handle the training loop
    time_start = time.time()
    trainer = pl.Trainer(max_epochs=e.EPOCHS, logger=False)
    trainer.fit(model, data_loader_train, data_loader_val)
        
    time_end = model.model_restorer.best_time
    e['train_time/gatv2'] = time_end - time_start
        
    model.eval()
        
    # Return the trained model
    return model


@experiment.hook('predict_model', replace=True, default=False)
def predict_model(e: Experiment,
                  index_data_map: dict,
                  model: Any,
                  indices: list[int],
                  ) -> np.ndarray:
    
    graphs = [index_data_map[i] for i in indices]
    data_list: List[Data] = pyg_data_list_from_graphs(graphs)
    data_loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    y_pred = []
    for data in data_loader:
        out = model(data).detach().cpu().numpy()
        y_pred.extend(out.tolist())
        
    y_pred = np.array(y_pred)
    return y_pred


@experiment.hook('predict_model_proba', replace=True, default=False)
def predict_model_proba(e: Experiment,
                        index_data_map: dict,
                        model: Any,
                        indices: list[int],
                        y_pred: np.ndarray,
                        ) -> np.ndarray:
    
    y_proba = softmax(y_pred, axis=1)
    return y_proba


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                    index_data_map: dict
                    ) -> None:
    
    for index, data in index_data_map.items():
        data['graph_features'] = np.zeros((e.CONV_UNITS[-1],))
        
        
@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict,
                  **kwargs
                  ) -> None:
    
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