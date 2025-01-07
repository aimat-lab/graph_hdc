import os
import time
import random
from typing import Any, List, Union

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict
from chem_mat_data.main import load_graph_dataset, load_dataset_metadata
from sklearn.preprocessing import StandardScaler

# :param IDENTIFIER:
#       String identifier that can be used to later on filter the experiment, for example.
IDENTIFIER: str = '1'

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'clintox'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'classification'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: Union[int, float] = 100
# :param NUM_VAL:
#       The number of validation samples to be used for the evaluation of the models during training.
NUM_VAL: int = 10
# :param SEED:
#       The random seed to be used for the experiment.
SEED: int = 1

# :param MODELS:
#       The list of models to be trained and evaluated. The models are trained and evaluated in the order they are
#       listed. The model names are dynamically evaluated as function names with the prefix 'train_model__{name}'.
#       if such a function exists in the experiment workspace, it is executed to train the model. The model is then
#       evaluated using the 'evaluate_model' function.
MODELS: List[str] = [
    'random_forest',
    'grad_boost',
    'k_neighbors',
    # 'gaussian_process',
    'neural_net',
    #'linear',
    'support_vector',
]


__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('load_dataset', replace=False, default=True)
def load_dataset(e: Experiment) -> dict[int, GraphDict]:
    
    # This function will download the dataset from the ChemMatData file share and return the already pre-processed 
    # list of graph dict representations.
    graphs: list[GraphDict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path='/tmp'
    )
    
    metadata = load_dataset_metadata(
        e.DATASET_NAME,
    )
    
    index_data_map = dict(enumerate(graphs))
    return index_data_map, metadata


@experiment.hook('get_graph_labels')
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    return graph['graph_labels']


@experiment.hook('filter_dataset', replace=False, default=True)
def filter_dataset(e: Experiment,
                   index_data_map: dict[int, dict],
                   ) -> tuple[list, list, list]:
    
    e.log('filtering dataset to remove invalid SMILES and unconnected graphs...')
    indices = list(index_data_map.keys())
    for index in indices:
        graph = index_data_map[index]
        smiles = graph['graph_repr']
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            del index_data_map[index]
            
        elif len(mol.GetAtoms()) < 2:
            del index_data_map[index]
            
        elif len(mol.GetBonds()) < 1:
            del index_data_map[index]


@experiment.hook('dataset_split', replace=False, default=True)
def dataset_split(e: Experiment,
                  indices: list[int],
                  ) -> tuple[list, list, list]:
    
    random.seed(e.SEED)
    
    # We accept NUM_TEST here to be either an integer or a float between 0 and 1
    # in case of an integer we use it as the number of test samples to be used
    # in case of a float we use it as the fraction of the dataset to be used as test samples.
    if isinstance(e.NUM_TEST, int):
        num_test = e.NUM_TEST
    elif isinstance(e.NUM_TEST, float):
        num_test = int(e.NUM_TEST * len(indices))
    
    test_indices = random.sample(indices, k=num_test)
    indices = list(set(indices) - set(test_indices))
    
    val_indices = random.sample(indices, k=e.NUM_VAL)
    indices = list(set(indices) - set(val_indices))
    
    train_indices = indices
    return train_indices, val_indices, test_indices
    
    
@experiment.hook('process_dataset', replace=False, default=True)
def process_dataset(e: Experiment,
                    index_data_map: dict
                    ) -> None:
    for index, graph in index_data_map.items():
        smiles: str = graph['graph_repr']
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fingerprint = gen.GetFingerprint(Chem.MolFromSmiles(smiles))
        graph['graph_features'] = np.array(fingerprint).astype(float)
    

@experiment.hook('after_dataset', replace=False, default=True)
def after_dataset(e: Experiment,
                  index_data_map: dict,
                  train_indices: list[int],
                  test_indices: list[int],
                  val_indices: list[int],
                  **kwargs,
                  ) -> None:
    
    # Plotting the histogram of graph sizes
    e.log('plotting histogram of graph sizes...')
    graph_sizes = [len(index_data_map[i]['node_indices']) for i in index_data_map.keys()]

    mean_size = np.mean(graph_sizes)
    p10_size = np.percentile(graph_sizes, 10)
    p90_size = np.percentile(graph_sizes, 90)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(graph_sizes, bins=30, ax=ax)
    ax.axvline(mean_size, color='black', linestyle='-', label=f'Mean: {mean_size:.2f}')
    ax.axvline(p10_size, color='black', linestyle='--', label=f'10th Percentile: {p10_size:.2f}')
    ax.axvline(p90_size, color='black', linestyle='--', label=f'90th Percentile: {p90_size:.2f}')
    ax.legend()
    ax.set_title(f'Histogram of Graph Sizes')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Count')
    e.commit_fig('graph_size_histogram.png', fig)
    
    if e.DATASET_TYPE == 'classification':
        
        # ~ plotting the label distribution
        
        e.log('plotting label distribution...')
        labels = np.array([np.argmax(index_data_map[i]['graph_labels']) for i in train_indices])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=labels.flatten(), ax=ax)
        ax.set_title('Label Distribution')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Count')
        e.commit_fig('label_distribution.png', fig)

    elif e.DATASET_TYPE == 'regression':
        
        # ~ plotting the value distribution
        
        e.log('plotting value distribution...')
        values = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(values.flatten(), ax=ax)
        ax.set_title('Value Distribution')
        ax.set_xlabel('Values')
        ax.set_ylabel('Count')
        e.commit_fig('value_distribution.png', fig)


@experiment.hook('train_model__random_forest', replace=False, default=True)
def train_model__random_forest(e: Experiment,
                               index_data_map: dict,
                               train_indices: list[int],
                               val_indices: list[int],
                               ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_estimators': 50,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': e.SEED,
    }
    
    if e.DATASET_TYPE == 'classification':
        model = MultiOutputClassifier(RandomForestClassifier(**kwargs))
        model.fit(X_train, y_train)
        
        return model
    
    elif e.DATASET_TYPE == 'regression':
        model = RandomForestRegressor(**kwargs)
        model.fit(X_train, y_train)
        
        return model
    
    
@experiment.hook('train_model__grad_boost', replace=False, default=True)
def train_model__grad_boost(e: Experiment,
                            index_data_map: dict,
                            train_indices: list[int],
                            val_indices: list[int]
                            ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'max_features': 'sqrt',
        'random_state': e.SEED,
    }
    
    if e.DATASET_TYPE == 'classification':
        model = ClassifierChain(GradientBoostingClassifier(
            **kwargs,
        ))
        model.fit(X_train, y_train)
        
        return model
    
    if e.DATASET_TYPE == 'regression':
        model = GradientBoostingRegressor(
            **kwargs,
        )
        model.fit(X_train, y_train)
        
        return model

    
@experiment.hook('train_model__k_neighbors', replace=False, default=True)
def train_model__k_neighbors(e: Experiment,
                             index_data_map: dict,
                             train_indices: list[int],
                             val_indices: list[int]
                             ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_neighbors': 7,
    }
    
    if e.DATASET_TYPE == 'classification':
        model = MultiOutputClassifier(KNeighborsClassifier(**kwargs))
        model.fit(X_train, y_train)
        
        return model
    
    elif e.DATASET_TYPE == 'regression':
        model = KNeighborsRegressor(**kwargs)
        model.fit(X_train, y_train)
        
        return model
    
@experiment.hook('train_model__gaussian_process', replace=False, default=True)
def train_model__gaussian_process(e: Experiment,
                                  index_data_map: dict,
                                  train_indices: list[int],
                                  val_indices: list[int]
                                  ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_restarts_optimizer': 3,
    }
    
    if e.DATASET_TYPE == 'classification':
        model = MultiOutputClassifier(GaussianProcessClassifier(**kwargs))
        model.fit(X_train, y_train)
        
        return model
    
    elif e.DATASET_TYPE == 'regression':
        model = GaussianProcessRegressor(**kwargs)
        model.fit(X_train, y_train)
        
        return model
    
    
@experiment.hook('train_model__neural_net', replace=False, default=True)    
def train_model__neural_net(e: Experiment,
                            index_data_map: dict,
                            train_indices: list[int],
                            val_indices: list
                            ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'hidden_layer_sizes': (100, 50, 25),
        'max_iter': 250,
        'alpha': 0.1,
    }
    
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(MLPClassifier(**kwargs))
        model.fit(X_train, y_train)
        return model
    
    elif e.DATASET_TYPE == 'regression':
        
        model = MLPRegressor(**kwargs)
        model.fit(X_train, y_train)
        return model
    

@experiment.hook('train_model__linear', replace=False, default=True)
def train_model__linear(e: Experiment,
                        index_data_map: dict,
                        train_indices: list[int],
                        val_indices: list
                        ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(LogisticRegression())
        model.fit(X_train, y_train)
        return model
    
    elif e.DATASET_TYPE == 'regression':
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model


@experiment.hook('train_model__support_vector', replace=False, default=True)
def train_model__support_vector(e: Experiment,
                                index_data_map: dict,
                                train_indices: list[int],
                                val_indices: list
                                ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'C': 1.0,
        'kernel': 'rbf',
        'max_iter': 250,
    }
    
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(SVC(**kwargs))
        model.fit(X_train, y_train)
        return model
    
    elif e.DATASET_TYPE == 'regression':
        
        model = SVR(**kwargs)
        model.fit(X_train, y_train)
        return model

    
@experiment.hook('evaluate_model', replace=False, default=True)
def evaluate_model(e: Experiment,
                   index_data_map: dict,
                   indices: list[int],
                   model: Any,
                   key: str,
                   scaler: Any = None,
                   **kwargs,
                   ) -> None:

    X_eval = np.array([index_data_map[i]['graph_features'] for i in indices])
    y_eval = np.array([index_data_map[i]['graph_labels'] for i in indices])
    
    y_pred = model.predict(X_eval)
    
    if e.DATASET_TYPE == 'classification':
        
        labels_pred = np.array([np.argmax(y) for y in y_pred])
        labels_eval = np.array([np.argmax(y) for y in y_eval])
        
        # ~ simple metrics
        acc_value = accuracy_score(y_eval, y_pred)
        f1_value = f1_score(y_eval, y_pred, average='macro')
        ap_value = average_precision_score(y_eval, y_pred, average='macro')
        e[f'metrics/{key}/acc'] = acc_value
        e[f'metrics/{key}/f1'] = f1_value
        e[f'metrics/{key}/ap'] = ap_value

        e.log(f' * accuracy: {acc_value:.3f}'
              f' - f1 (macro): {f1_value:.3f}'
              f' - ap (macro): {ap_value:.3f}')

        # ~ confusion matrix
        cm = confusion_matrix(labels_eval, labels_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix\n'
                     f'Accuracy: {acc_value:.3f} - F1 (macro): {f1_value:.3f} - Average Precision (macro): {ap_value:.3f}')
        e.commit_fig(f'{key}__confusion_matrix.png', fig)
        
    elif e.DATASET_TYPE == 'regression':
        
        if scaler:
            y_eval = scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # ~ simple metrics
        r2_value = r2_score(y_eval, y_pred)
        mse_value = mean_squared_error(y_eval, y_pred)
        mae_value = mean_absolute_error(y_eval, y_pred)
        e[f'metrics/{key}/r2'] = r2_value
        e[f'metrics/{key}/mse'] = mse_value
        e[f'metrics/{key}/mae'] = mae_value
        
        e.log(f' * r2: {r2_value:.3f}'
              f' - mse: {mse_value:.3f}'
              f' - mae: {mae_value:.3f}')
        
        # ~ plotting the regression plots
        fig, ax = plt.subplots(figsize=(10, 8))
        df = pd.DataFrame({
            'y_true': y_eval.flatten(),
            'y_pred': y_pred.flatten()
        })
        max_value = max(df['y_true'].max(), df['y_pred'].max())
        min_value = min(df['y_true'].min(), df['y_pred'].min())
        ax.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='-', alpha=0.5)
        sns.histplot(
            df, 
            x='y_true', 
            y='y_pred', 
            ax=ax,
            bins=50,
            cbar=True,
            binrange=(min_value, max_value)
        )
        ax.set_title(f'Regression Plot\n'
                     f'R2: {r2_value:.3f} - MSE: {mse_value:.3f} - MAE: {mae_value:.3f}')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        
        plt.tight_layout()
        e.commit_fig(f'{key}__regression_plots.png', fig)
        

@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment to predict molecule dataset...')
    
    # ~ data loading
    
    # This hook returns a dict whose keys are the unique integer indices of the dataset elements and the values 
    # are the corresponding graph dict representations.
    e.log(f'loading dataset "{e.DATASET_NAME}"...')
    index_data_map: dict[int, GraphDict]
    index_data_map, metadata = e.apply_hook(
        'load_dataset',
    )
    
    e.log('determine the graph labels...')
    for index, graph in index_data_map.items():
        # :hook get_graph_labels:
        #       This hook is called on each graph in the dataset and is supposed to return the numpy array 
        #       representing the graph labels to serve as the prediction target.
        graph['graph_labels'] = e.apply_hook(
            'get_graph_labels',
            index=index,
            graph=graph
        )
    
    # :hook filter_dataset:
    #       An action hook that is called after the dataset has been loaded and before the dataset indices are 
    #       obtained, this optional hook presents the opportunity to filter the dataset based on certain criteria.
    e.apply_hook(
        'filter_dataset',
        index_data_map=index_data_map,
    )
    
    indices = list(index_data_map.keys())
    e.log(f'loaded dataset with {len(index_data_map)} elements...')
    
    e.log('creating train-val-test split...')
    train_indices, val_indices, test_indices = e.apply_hook(
        'dataset_split',
        indices=indices
    )
    e.log(f'train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}')
    e['indices/train'] = train_indices
    e['indices/val'] = val_indices
    e['indices/test'] = test_indices
    
    # ~ dataset processing
    e.log('processing dataset...')
    time_start = time.time()
    e.apply_hook(
        'process_dataset',
        index_data_map=index_data_map
    )
    time_end = time.time()
    duration = time_end - time_start
    e.log(f'processed dataset after {duration:.2f} seconds')
    
    # ~ scaling output
    
    scaler = None
    if e.DATASET_TYPE == 'regression':
        scaler = StandardScaler()
        y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
        scaler.fit(y_train)

        for index in index_data_map:
            index_data_map[index]['graph_labels'] = scaler.transform(index_data_map[index]['graph_labels'].reshape(1, -1)).flatten()
    
    # :hook after_dataset:
    #       An action hook that is called after the dataset has been loaded and processed. This hook
    #       presents the opportunity to perform additional processing on the dataset before training
    #       the models.
    e.apply_hook(
        'after_dataset',
        index_data_map=index_data_map,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    
    example_graph = index_data_map[train_indices[0]]
    e.log(f'example graph'
          f' - num_nodes: {len(example_graph["node_indices"])}'
          f' - num edges: {len(example_graph["edge_indices"])}'
          f' - embedding shape: {example_graph["graph_features"].shape}')
    
    # ~ model training
    for model_name in e.MODELS:
        
        e.log(f'\ntraining model "{model_name}"...')
        time_start = time.time()
        model = e.apply_hook(
            f'train_model__{model_name}',
            index_data_map=index_data_map,
            train_indices=train_indices,
            val_indices=val_indices,
        )
        time_end = time.time()
        duration = time_end - time_start
        e.log(f'training done after {duration:.2f} seconds')
        
        # ~ model evaluation
        e.log('evaluating model...')
        e.apply_hook(
            'evaluate_model',
            model=model,
            index_data_map=index_data_map,
            indices=test_indices,
            key=f'test_{model_name}',
            scaler=scaler,
        )
        
    # ~ comparison of models
    
    e.log('creating model comparison plots...')
    keys = list(e['metrics'].keys())
    metrics = list(e['metrics'][keys[0]].keys())
    
    for metric in metrics:
        
        fig, ax = plt.subplots(figsize=(10, 8))
        values = [e['metrics'][key][metric] for key in keys]
        sns.barplot(x=keys, y=values, ax=ax)
        ax.set_title(f'Comparison of {metric}')
        ax.set_xlabel('Models')
        ax.set_ylabel(metric)
        ax.set_xticklabels(keys, rotation=45, ha='right')
        plt.tight_layout()
        e.commit_fig(f'comparison_{metric}.png', fig)
    

experiment.run_if_main()
    