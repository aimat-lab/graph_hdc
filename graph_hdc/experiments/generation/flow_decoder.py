import itertools
import random
import copy
from typing import Generator, Union, List

import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import polars as pol
import pytorch_lightning as pl
from torch.utils.data import (
    IterableDataset,
    get_worker_info
)
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.utils import scatter

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

# graph_hdc
from graph_hdc.special.molecules import (
    graph_dict_from_mol,
    make_molecule_node_encoder_map,
    nx_from_graph_dict,
)
from graph_hdc.graph import (
    data_from_graph_dict,
    data_add_full_connectivity,
)
from graph_hdc.models import HyperNet
from graph_hdc.graph import (
    data_from_graph_dict,
    data_list_from_graph_dicts,
)


# Atom colors mapping (atomic number -> color)
ATOM_COLORS = {
    1: 'white',      # Hydrogen
    6: 'gray',       # Carbon
    7: 'blue',       # Nitrogen
    8: 'red',        # Oxygen
    9: 'lightgreen', # Fluorine
    15: 'orange',    # Phosphorus
    16: 'yellow',    # Sulfur
    17: 'green',     # Chlorine
    35: 'brown',     # Bromine
    53: 'purple',    # Iodine
}

# Atom symbol mapping (atomic number -> symbol)
ATOM_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def plot_molecular_graph(graph: dict, ax, smiles: str, graph_idx: int, similarity: float):
    """
    Plot a molecular graph with proper atom labels and colors.

    :param graph: Graph dictionary containing molecular structure
    :param ax: Matplotlib axis to plot on
    :param smiles: SMILES string for the molecule
    :param graph_idx: Index of the decoded graph
    :param similarity: Similarity score to the original
    """
    try:
        # Convert graph dict to NetworkX graph
        G = nx_from_graph_dict(graph)

        # Extract atom types from node_atoms
        node_atoms = graph.get('node_atoms', [])

        # Create node labels and colors based on atomic numbers
        labels = {}
        node_colors = []

        for i, atomic_num in enumerate(node_atoms):
            labels[i] = ATOM_SYMBOLS.get(atomic_num, str(atomic_num))
            node_colors.append(ATOM_COLORS.get(atomic_num, 'lightgray'))

        # Create layout and draw graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax,
                labels=labels,
                node_color=node_colors,
                node_size=300,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color='gray')

        ax.set_title(f'SMILES: {smiles}\nDecoded Graph {graph_idx+1}\nSimilarity: {similarity:.3f}')
        ax.axis('off')

    except Exception as ex:
        ax.text(0.5, 0.5, f'Error visualizing\ngraph {graph_idx+1}:\n{str(ex)}',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'SMILES: {smiles}\nDecoded Graph {graph_idx+1} (Error)')
        ax.axis('off')


DATASET_PATH: str = '/media/ssd2/Programming/graph_hdc/graph_hdc/experiments/generation/qm9_smiles.csv'
SEED: int = 0

# === HDC PARAMETERS ===

EMBEDDING_SIZE: int = 15_000
NUM_LAYERS: int = 2

# === MODEL PARAMETERS ===


class SmilesGraphDataset(IterableDataset):
    
    def __init__(
        self,
        dataset: str,
        encoder: HyperNet,
        smiles_column: str = 'smiles',
        reservoir_sampling: bool = True,
        reservoir_size: int = 1000,
    ) -> None:
        
        super().__init__()
        
        self.dataset = dataset
        #self.encoder = encoder
        self.smiles_column = smiles_column
        self.reservoir_sampling = reservoir_sampling
        self.reservoir_size = reservoir_size
        
        # --- reservoir sampling setup ---
        # If reservoir sampling is enabled, we need to set up a reservoir to hold the samples
        # from which we then randomly draw during iteration.
        self.reservoir: list[dict] = []
        
    def _create_frame(self) -> pol.LazyFrame:
        
        if isinstance(self.dataset, str):
            df = pol.read_csv(self.dataset).lazy()
            
        return df
        
    def __iter__(self) -> Generator[None, None, torch.Tensor]:
        
        # --- creating the frame ---
        # The self._create_frame method actually intializes the polars lazyframe from whatever 
        # data source was given in the constructor of the dataset class. This lazy frame can then 
        # be used to stream the data from that source. It's important that we do this here in 
        # the __iter__ method so that in a multi-worker setting each worker creates its own
        # frame and does not share it with other workers - which would cause a deadlock trying 
        # to access the same IO object.
        worker_info = get_worker_info()
        lazyframe: pol.LazyFrame = self._create_frame()

        # For a single worker it is very easy we just stream the data from the lazyframe as it 
        # is.
        if worker_info is None:
            dataframe: pol.DataFrame = lazyframe.collect(streaming=True)
        # Otherwise we have to assign which elements of the original data to associate with 
        # which worker!
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            dataframe: pol.DataFrame = (
                lazyframe
                .with_row_index('__row_idx')
                .filter((pl.col('__row_idx') % num_workers) == worker_id)
                .drop('__row_idx')
                .collect(streaming=True)
            )
            
        # --- iterating over the dataframe ---
        # Now that we have a streamed dataframe instance we can iterate over the rows in this 
        # dataframe to get the individual smiles strings that it is made up of.
        for row in dataframe.iter_rows(named=True):
            
            # --- reservoir sampling ---
            # If reservoir sampling is enabled we need to add the current row to the reservoir
            # and then randomly draw from the reservoir to return a sample.
            if self.reservoir_sampling:
                
                # Handle reservoir sampling
                # As long as the reservoir is not full we just add to the reservoir. Once it 
                # is fully we replace a random element from it with the new element of the 
                # current iteration.
                if len(self.reservoir) < self.reservoir_size:
                    self.reservoir.append(row)
                else:
                    # Replace a random element
                    idx = random.randint(0, self.reservoir_size - 1)
                    self.reservoir[idx] = row

                # Randomly select an element from reservoir to emulate batching.
                if len(self.reservoir) > 0:
                    idx = random.choice(range(len(self.reservoir)))
                    row = self.reservoir[idx]
                else:
                    continue
                
                sample = row
                
            else:
                sample = row
            
            # --- processing to graph ---
            # With the selected SMILES we can now process that into a graph representation
            # that can be used for the model.
            try:
                
                smiles: str = sample[self.smiles_column]
                mol = Chem.MolFromSmiles(smiles)
                # If the current SMILES is invald we just skip this element and go to the 
                # next one.
                if not mol:
                    continue
                
                if len(mol.GetAtoms()) < 2:
                    continue
                
                # Then we first convert that mol object into a graph dict representation.
                # This is a dictionary that contains all the information to represent the 
                # molecule as a graph, including the information about the atom types, 
                # bond types, connectivity, etc.
                graph: dict = graph_dict_from_mol(mol)
                
                # This can then be converted into a PyG Data object which is the thing we 
                # ultimately need for the training of the graph neural network later on.
                data = data_from_graph_dict(graph)
                
                # This method will add two additional properties dynamically to the Data object
                # data.edge_index_full: Complete edge list representing full connectivity (2, num_nodes**2)
                # data.edge_weight_full: Binary weights indicating which edges exist in original graph 
                # (num_nodes**2,)
                data = data_add_full_connectivity(data)
                
                # with torch.no_grad():
                #     results = self.encoder.forward(data)
                #     cond = results['graph_embedding']
                    
                # data.cond = cond
                
                yield data
                
            except Exception as exc:
                continue


class EdgeReconstructionFlow(pl.LightningModule):
    
    def __init__(
        self,
        encoder: HyperNet,
        time_dim: int = 1,
        feat_dim: int = 1,
        cond_dim: int = 128,
        hidden_dim: int = 128,
        conv_units: int = [128, 128, 128],
        learning_rate: float = 1e-3,
        epsilon: float = 0.0001,
    ) -> None:
        
        super().__init__()
        
        self.encoder = encoder
        self.time_dim = time_dim
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        self.conv_units = conv_units
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.lay_emb_node = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.lay_emb_time = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.lay_emb_cond = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        prev_units = hidden_dim * 3
        for units in conv_units:
            lay_conv = GATv2Conv(
                in_channels=prev_units,
                out_channels=units,
                edge_dim=1,
                heads=5,
                concat=False,
                add_self_loops=False,
            )
            self.conv_layers.append(lay_conv)
            
            self.norm_layers.append(nn.BatchNorm1d(units))
            
            prev_units = units
        
        self.lay_pred = nn.Sequential(
            nn.Linear(prev_units * 2, prev_units),
            nn.BatchNorm1d(prev_units),
            nn.LeakyReLU(),
            nn.Linear(prev_units, prev_units),
            nn.BatchNorm1d(prev_units),
            nn.LeakyReLU(),
            nn.Linear(prev_units, 2)
        )
        
        # --- flow matching stuff ---
        
        self.scheduler = PolynomialConvexScheduler(n=2)
        self.path = MixtureDiscreteProbPath(scheduler=self.scheduler)
        
        self.criterion = MixturePathGeneralizedKL(self.path)

        
    def forward(
        self,
        data: Data,
        edge_weight: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        
        # --- embedding ---
        # First we need to embed the node features, the time and the conditioning information
        # into a common space that can then be processed by the GNN.
        node_features = torch.cat([
            data.node_atoms.unsqueeze(-1),
            data.node_degrees.unsqueeze(-1),
            data.node_valences.unsqueeze(-1),
        ], dim=-1)
        h_node = self.lay_emb_node(node_features)
        h_time = self.lay_emb_time(t.unsqueeze(-1))
        h_cond = self.lay_emb_cond(cond)
        
        # We need to expand the time and conditioning embeddings to match the number of nodes
        # in the graph.
        h_cond = h_cond[data.batch]
        
        # Then we concatenate all these embeddings together to form the initial node 
        # representations that will be fed into the GNN.
        h = torch.cat([h_node, h_time, h_cond], dim=-1)
        
        # --- GNN processing ---
        # Now we can process these node representations through the GNN layers.
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            h = conv(h, data.edge_index_full, edge_weight.float())
            h = norm(h)
            h = F.leaky_relu(h)
        
        # --- edge prediction ---
        # Finally we need to predict the edge values (i.e., the presence or absence of edges)
        # based on the final node representations from the GNN.
        
        # To do this we first need to gather the node representations for each edge in the 
        # graph. We do this by indexing into the node representations using the edge index.
        row, col = data.edge_index_full
        edge_h = torch.cat([h[row], h[col]], dim=-1)
        
        # Then we pass these edge representations through a final prediction layer to get 
        # the edge scores.
        edge_scores = self.lay_pred(edge_h).squeeze(-1)
        
        # We return a new Data object that contains the original graph structure along with 
        # the predicted edge scores.
        return edge_scores
    
    def sample_prior(self, data: Data) -> torch.Tensor:
        
        base_sparsity = 0.2
        
        edge_probs = torch.full_like(data.edge_weight_full, base_sparsity)
        edge_probs = edge_probs.to(self.device)
        binary = torch.bernoulli(edge_probs).squeeze(-1)
        
        return binary
    
    def sample_prior_ot(self, data: Data, k: int = 32, sparsity: float = 0.2) -> torch.Tensor:
        
        batch_edge = data.batch[data.edge_index_full[0]]
        
        E = data.edge_weight_full.size(0)
        edge_probs = torch.full((E, k), sparsity).to(self.device)
        edge_binary = torch.bernoulli(edge_probs)
    
        # (B, k)
        dist = scatter(
            (edge_binary != data.edge_weight_full).float(), 
            index=batch_edge,
            dim=0,
            reduce='sum',
        )
        dist_edge = dist[batch_edge]
        # (E, )
        indices = torch.argmin(dist_edge, dim=1)

        edge_opt = torch.gather(edge_binary, 1, indices.unsqueeze(-1)).squeeze(-1)
        return edge_opt
    
    def training_step(
        self,
        batch: Data,
        batch_idx: int
    ) -> torch.Tensor:
        
        batch_size = batch.batch.max().item() + 1
        
        # --- condition ---
        # The condition for the denoising is the HyperNet embedding of the graph structure
        # that we want to reconstruct the edges for so here we first pass the batch through
        # the hypernet to get these embeddings
        with torch.no_grad():
            results = self.encoder.forward(batch)
            cond = results['graph_embedding']
        
        # sample time
        t_graph = (torch.rand((batch_size, )) * (1.0 - self.epsilon)).to(self.device)
        # now we need to cast the batch size onto the edge dimension
        t_node = t_graph[batch.batch]
        t_edge = t_node[batch.edge_index_full[0]]
        
        # sample prior
        edge_0 = self.sample_prior_ot(batch)
        
        # construct path
        path_sample = self.path.sample(t=t_edge, x_0=edge_0, x_1=batch.edge_weight_full.squeeze(-1))
        
        # predict velocity field
        pred = self.forward(
            batch,
            edge_weight=path_sample.x_t,
            t=t_node,
            cond=cond,
        )
        
        # calculate loss
        loss = self.criterion(
            logits=pred,
            x_t=path_sample.x_t.long(),
            x_1=batch.edge_weight_full.squeeze().long(),
            t=path_sample.t,
        )
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate
        )
        return optimizer



class EdgeReconstructionWrapper(ModelWrapper):
    
    def forward(self, x: torch.tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        
        data = extras['data']
        
        logits = self.model(
            data=extras['data'],
            t=t[0].expand(data.x.size(0)), 
            cond=extras['cond'], 
            edge_weight=x,
        )
        probs = torch.softmax(logits, dim=-1)
        return probs
    
    def decode(
        self, 
        hv: torch.Tensor,
        batch_size: int = 16, 
        time_steps: int = 250
    ) -> torch.Tensor:
        
        # make sure that hv is a torch tensor
        if not isinstance(hv, torch.Tensor):
            hv = torch.tensor(hv, dtype=torch.float32)
        
        # --- extracting the nodes ---
        # First of all we need to extract the nodes from the hypervector embeddings.
        
        graph: dict = self.model.encoder.decode_nodes(hv)
        graph['node_attributes'] = graph['node_atoms']
        data: Data = data_from_graph_dict(graph)
        
        batch = next(iter(DataLoader([data] * batch_size, batch_size=batch_size)))
        
        # --- sampling from prior ---
        
        x_init = self.model.sample_prior(batch)
        
        # --- sampling edges from flow ---
        
        solver = MixtureDiscreteEulerSolver(model=self, path=self.model.path, vocabulary_size=2)
        
        linspace = torch.linspace(0, 1.0 - self.model.epsilon, time_steps)
        edges_sampled = solver.sample(
            x_init=x_init.long(),
            step_size=1/time_steps,
            time_grid=linspace.to(self.model.device),
            # model extras
            data=batch,
            cond=torch.stack([hv] * batch_size, dim=0),
        )
        
        print(edges_sampled)
        
        # --- constructing graphs ---
        
        edges_sampled = (edges_sampled > 0.5).float()
        edges_sampled = edges_sampled.reshape(batch_size, -1)
        
        graphs_decoded = []
        for i, (edge_weights) in enumerate(edges_sampled):
            
            graph_decoded = copy.deepcopy(graph)
            
            edge_index = []
            for (src, dst), weight in zip(graph['edge_index_full'], edge_weights):
                
                if weight > 0.5:
                    edge_index.append([src, dst])
                
            if not edge_index:
                edge_index = np.array([[0, 0]])
                
            graph_decoded['edge_indices'] = np.array(edge_index)
            #graph_decoded['edge_index'] = np.array(edge_index).T
            graphs_decoded.append(graph_decoded)
            
        # --- sorting graphs ---
        # At the very end we want to encode all of the decoded graphs with the encoder and
        # sort them by their cosine similarity to the original hypervector.

        similarities = []
        for i, graph_decoded in enumerate(graphs_decoded):
            try:
                # Encode the decoded graph
                results = self.model.encoder.forward_graph(graph_decoded)
                graph_hv = torch.tensor(results['graph_embedding'])

                # Compute cosine similarity with original hypervector
                similarity = F.cosine_similarity(
                    hv.unsqueeze(0),
                    graph_hv.unsqueeze(0),
                    dim=1
                ).item()
                similarities.append((i, similarity))

            except Exception as e:
                # If encoding fails, assign low similarity
                similarities.append((i, -1.0))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Reorder graphs_decoded based on similarity scores
        graphs_decoded = [graphs_decoded[i] for i, _ in similarities]
        similarity_values = [sim for _, sim in similarities]

        return graphs_decoded, similarity_values
            




__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment
def experiment(e: Experiment):
    
    e.log('starting the experiment...')
    e.log_parameters()
    
    # --- creating hypernet encoder ---
    
    e.log('creating the HDC HyperNet encoder...')
    
    node_encoder_map = make_molecule_node_encoder_map(
        dim=e.EMBEDDING_SIZE,
        seed=e.SEED,
    )
    
    encoder = HyperNet(
        hidden_dim=e.EMBEDDING_SIZE,
        depth=e.NUM_LAYERS,
        node_encoder_map=node_encoder_map,
        seed=e.SEED,
    )
    
    # --- creating dataset ---
    
    e.log('initializing dataset...')
    dataset = SmilesGraphDataset(
        dataset=e.DATASET_PATH,
        encoder=encoder,
        smiles_column='smiles',
        reservoir_sampling=True,
        reservoir_size=5000,
    )   
    
    e.log('sanity checking the encoder by iterating some elements...')
    for i, data in enumerate(itertools.islice(iter(DataLoader(dataset, batch_size=16)), 5)):
        e.log(
            f' * Data {i}'
            f' - {torch.max(data.batch) + 1} graphs'
            f' - {data.node_atoms.size(0)} nodes'
            f' - {data.edge_index.size(1)} edges'
            f' - {data.edge_index_full.size(1)} full edges'
            f' - {data.edge_weight_full.size(0)} edge weights'
        )
        results = encoder.forward(data)
        e.log(f'   - embedding: {results["graph_embedding"].size()}')
        
    # --- model training ---
        
    e.log('creating the flow model...')
    model = EdgeReconstructionFlow(
        encoder=encoder,
        time_dim=1,
        feat_dim=1,
        cond_dim=e.EMBEDDING_SIZE,
        hidden_dim=128,
        conv_units=[256, 256, 256],
        learning_rate=1e-3,
        epsilon=0.001,
    )
        
    loader_train = DataLoader(
        dataset,
        batch_size=64,
        #num_workers=4,
        #prefetch_factor=2,
    )
    
    trainer = pl.Trainer(
        max_epochs=3,
        #accelerator='cpu',
        accelerator='auto',
        devices='auto',
        enable_progress_bar=True,
        logger=False,
        enable_model_summary=True,
        limit_train_batches=1000,  # for debugging
    )
    
    e.log('starting the training...')
    trainer.fit(
        model,
        loader_train,
    )
    model.eval()

    
    # --- evaluating the model ---
    
    e.log('evaluating the model...')
    wrapper = EdgeReconstructionWrapper(model)
    
    SMILES_LIST = [
        'C1=CC=CC=C1CC(CO)CN',
        'CCC(CC(F)(F)F)CCCCN',
        'C1=CC=CC=C1',
        'OCCCOC(=N)CO',
        'C(CO)COC(=O)CO',
        'C(COCC(=O)CO)O',
        'CC(C)(C)CCCCO',
        'CC(C)(C)COCCO',
        'CC(C)(C)OCCCO',
        'CC(C)(CCCCO)O',
    ]
    
    # Create figure for all decoded graphs
    num_molecules = len(SMILES_LIST)
    max_decoded_per_mol = 4  # Limit to first 4 decoded graphs per molecule

    fig, axes = plt.subplots(num_molecules, max_decoded_per_mol,
                            figsize=(16, 4 * num_molecules))

    if num_molecules == 1:
        axes = axes.reshape(1, -1)

    for mol_idx, smiles in enumerate(SMILES_LIST):

        e.log(f'evaluating the model on {smiles}...')
        results: dict = encoder.forward_graph(graph_dict_from_mol(Chem.MolFromSmiles(smiles)))
        hv = results['graph_embedding']

        graphs_decoded, similarity_values = wrapper.decode(
            hv,
            batch_size=32,
        )

        for i, graph in enumerate(graphs_decoded[:max_decoded_per_mol]):
            similarity = similarity_values[i] if i < len(similarity_values) else -1.0
            plot_molecular_graph(graph, axes[mol_idx, i], smiles, i, similarity)

        # Fill remaining subplots if fewer decoded graphs than max
        for i in range(len(graphs_decoded), max_decoded_per_mol):
            ax = axes[mol_idx, i]
            ax.text(0.5, 0.5, 'No decoded graph',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'SMILES: {smiles}\nDecoded Graph {i+1} (Empty)')
            ax.axis('off')

    plt.tight_layout()
    e.commit_fig('decoded_graphs_visualization', fig)

experiment.run_if_main()