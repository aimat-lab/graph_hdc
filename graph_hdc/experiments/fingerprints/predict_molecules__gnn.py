import os
import time
import random
from typing import List, Any, Literal, Optional

import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from scipy.special import softmax
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from torch_geometric.loader import DataLoader

from graph_hdc.baselines.gnn import GNN_CLASSES
from graph_hdc.baselines.gnn import hdf_matched_graph
from graph_hdc.baselines.gnn import build_pyg_list


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

# :param NODE_FEATURES:
#       Which node features the GNN receives. 'default' uses the full ChemMatData featurization of the
#       dataset (element, hybridization, degree, H count, charge, aromaticity, ring membership, mass, Crippen
#       contributions). 'hdf' restricts the input to exactly the atom attributes that the hyperdimensional
#       fingerprint encodes (element, heavy-atom degree, implicit H count; no bond types), which makes the
#       GNN comparison a test of the encoding mechanism rather than of the input features.
NODE_FEATURES: Literal['default', 'hdf'] = 'default'
# :param CONV_UNITS:
#       A list of integers specifying the number of units in each convolutional layer of the GNN models.
CONV_UNITS: List[int] = [128, 128, 128]
# :param DENSE_UNITS:
#       A list of integers specifying the number of units in each dense (fully connected) layer of the GNN models.
DENSE_UNITS: List[int] = [128, 64, 32]
# :param BATCH_SIZE:
#       The size of the batches to be used during training. This parameter determines the number of samples
#       that are processed in parallel during the training of the model.
BATCH_SIZE: int = 32
# :param EPOCHS:
#       The maximum number of training epochs. With early stopping enabled this is only an upper bound.
EPOCHS: int = 200
# :param EARLY_STOPPING_PATIENCE:
#       Training stops once the validation metric has not improved for this many epochs. In any case, the
#       weights of the epoch with the best validation metric are restored at the end of training. None
#       (the default, which keeps older configs unchanged) always trains for the full number of EPOCHS;
#       the GNN comparison (ex_14) uses 50 with EPOCHS=1000.
EARLY_STOPPING_PATIENCE: Optional[int] = None
# :param LEARNING_RATE:
#       The learning rate to be used for the training of the model. This parameter determines the step size that
#       is used to update the model parameters during training.
LEARNING_RATE: float = 1e-4

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


def train_gnn(e: Experiment,
              name: str,
              index_data_map: dict,
              train_indices: list[int],
              ) -> Any:
    """
    Train the GNN of type ``name`` ('gcn', 'gin' or 'gatv2') end-to-end on the prediction target.

    Like the ``neural_net2`` baseline, 5% of the training indices are held out as an internal validation
    set. The weights of the epoch with the best validation metric are restored at the end, and training
    stops early once that metric has not improved for EARLY_STOPPING_PATIENCE epochs. The training time
    (up to the best epoch), the best epoch and the total number of epochs are recorded.
    """
    pl.seed_everything(e.SEED, workers=True)

    num_val = max(2, int(0.05 * len(train_indices)))
    val_indices_ = random.sample(train_indices, k=num_val)
    train_indices = list(set(train_indices) - set(val_indices_))

    # Get example graph for model initialization
    example_graph = index_data_map[train_indices[0]]

    data_loader_train = DataLoader(
        build_pyg_list(index_data_map, train_indices),
        batch_size=e.BATCH_SIZE,
        shuffle=True,
    )
    data_loader_val = DataLoader(
        build_pyg_list(index_data_map, val_indices_),
        batch_size=e.BATCH_SIZE,
        shuffle=False,
    )

    model = GNN_CLASSES[name](
        input_dim=example_graph['node_attributes'].shape[1],
        output_dim=example_graph['graph_labels'].shape[0],
        output_type=e.DATASET_TYPE,
        conv_units=e.CONV_UNITS,
        dense_units=e.DENSE_UNITS,
        learning_rate=e.LEARNING_RATE,
        early_stopping_patience=e.EARLY_STOPPING_PATIENCE,
    )

    # Use PyTorch Lightning's Trainer to handle the training loop. Default checkpointing is disabled
    # because parallel runs would race on the shared ./checkpoints folder; best-state selection is done
    # in memory by the BestModelRestorer callback instead.
    time_start = time.time()
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, data_loader_train, data_loader_val)

    # best_time is None only if the validation metric never improved (e.g. NaN); keep the run anyway
    best_time = model.model_restorer.best_time
    e[f'train_time/{name}'] = (best_time if best_time is not None else time.time()) - time_start
    e[f'best_epoch/{name}'] = model.model_restorer.best_epoch
    e[f'epochs/{name}'] = trainer.current_epoch
    # learning curve: (epoch, validation metric) for every epoch
    e[f'history/{name}'] = model.model_restorer.history
    e.log(f'trained {name} for {trainer.current_epoch} epochs, best epoch {model.model_restorer.best_epoch}')

    model.eval()

    # Return the trained model
    return model


@experiment.hook('train_model__gcn', replace=False, default=True)
def train_model__gcn(e: Experiment,
                     index_data_map: dict,
                     train_indices: list[int],
                     val_indices: list[int],
                     ) -> Any:
    """
    This hook is called during the training period of the experiment to train a model of the "gcn" type.
    """
    return train_gnn(e, 'gcn', index_data_map, train_indices)


@experiment.hook('train_model__gin', replace=False, default=True)
def train_model__gin(e: Experiment,
                     index_data_map: dict,
                     train_indices: list[int],
                     val_indices: list[int],
                     ) -> Any:
    """
    This hook is called during the training period of the experiment to train a model of the "gin" type.
    """
    return train_gnn(e, 'gin', index_data_map, train_indices)


@experiment.hook('train_model__gatv2', replace=False, default=True)
def train_model__gatv2(e: Experiment,
                       index_data_map: dict,
                       train_indices: list[int],
                       val_indices: list[int],
                       ) -> Any:
    """
    This hook is called during the training period of the experiment to train a model of the "gatv2" type.
    """
    return train_gnn(e, 'gatv2', index_data_map, train_indices)


@experiment.hook('predict_model', replace=True, default=False)
def predict_model(e: Experiment,
                  index_data_map: dict,
                  model: Any,
                  indices: list[int],
                  ) -> np.ndarray:

    model.eval()
    data_loader = DataLoader(
        build_pyg_list(index_data_map, indices),
        batch_size=e.BATCH_SIZE,
        shuffle=False,
    )
    y_pred = []
    with torch.no_grad():
        for data in data_loader:
            out = model(data.to(model.device)).cpu().numpy()
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
    """
    The GNNs operate on the graphs directly, so there is no fixed vector representation. The
    placeholder "graph_features" only exist because the base experiment expects them. If NODE_FEATURES
    is 'hdf', the node and edge features are replaced by the HDF-matched featurization.
    """
    for index, data in index_data_map.items():
        if e.NODE_FEATURES == 'hdf':
            hdf_matched_graph(data)
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