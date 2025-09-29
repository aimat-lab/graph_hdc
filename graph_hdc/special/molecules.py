from typing import List, Any, Union, Dict

import math
import torch
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import rdkit.Chem as Chem
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdmolops
from chem_mat_data.processing import MoleculeProcessing
from graph_hdc.utils import AbstractEncoder
from graph_hdc.utils import CategoricalIntegerEncoder
from graph_hdc.utils import ContinuousEncoder

pt = Chem.GetPeriodicTable()


class AtomEncoder(AbstractEncoder):
    
    periodic_table = GetPeriodicTable()
    
    def __init__(self,
                 dim: int,
                 atoms: List[Union[str, int]],
                 seed: int = 0,
                 ) -> None:
        AbstractEncoder.__init__(self, dim, seed)
        #self.periodic_table = GetPeriodicTable()
        
        self.dim = dim
        self.atoms: List[int] = [
            self.get_atomic_index(atom) if isinstance(atom, str) else atom
            for atom in atoms
        ]
        self.num_categories = len(self.atoms) + 1
        
        self.atom_index_map = {atom: i for i, atom in enumerate(self.atoms)}
        self.index_atom_map = {i: atom for i, atom in enumerate(self.atoms)}
        
        torch.manual_seed(seed)
        self.dist = torch.distributions.Normal(0.0, 1.0 / np.sqrt(dim))
        self.embeddings = self.dist.sample((self.num_categories, dim))
        # random = np.random.default_rng(seed)
        # self.embeddings: torch.Tensor = torch.tensor(random.normal(
        #     # This scaling is important to have normalized base vectors
        #     loc=0.0,
        #     scale=(1.0 / np.sqrt(self.dim)), 
        #     size=(self.num_categories, self.dim)
        # ).astype(np.float32))
        
    def get_atomic_index(self, atom: str) -> int:
        return self.periodic_table.GetAtomicNumber(atom)
        
    def get_atomic_symbol(self, index: int) -> str:
        return self.periodic_table.GetElementSymbol(index)
        
    def encode(self, atom: Union[int, str]) -> torch.Tensor:
        if isinstance(atom, str):
            atom = self.get_atomic_index(atom)
            
        atom = int(atom)
        if atom in self.atom_index_map:
            index = self.atom_index_map[int(atom)]
        # The last element in the embeddings tensor is the "unknown" case
        else:
            index = -1
        
        return self.embeddings[index]
    
    def decode(self, hv: torch.Tensor) -> Any:
        distances = [torch.norm(hv - embedding) for embedding in self.embeddings]
        closest_embedding_index = int(torch.argmin(torch.tensor(distances)))
        return self.index_atom_map[closest_embedding_index]
    
    def get_encoder_hv_dict(self):
        return dict(zip(self.atoms, self.embeddings))



def nx_from_graph_dict(graph: dict) -> nx.Graph:
    """
    Creates a new networkx Graph from the given graph dict representation.
    
    :param graph: The graph dict representation to be converted.
    
    :returns: A networkx Graph instance
    """
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, node_index in enumerate(graph['node_indices']):
        G.add_node(int(node_index))
        labels = []
        for key, value in graph.items():
            if key.startswith('node_'):
                val = value[i]
                # convert numpy scalar to python scalar if needed
                if isinstance(val, np.generic):
                    val = val.item()
                G.nodes[int(node_index)][key] = val
                labels.append(f"{key}={val}")
                
        # composite label with all node_* attributes, append " - the rest"
        G.nodes[int(node_index)]['label'] = (" - ".join(labels)) if labels else ""
    
    # Add edges
    for i, j in graph['edge_indices']:
        G.add_edge(int(i), int(j))
    
    return G



def graph_dict_from_mol(mol: Chem.Mol,
                        processing: MoleculeProcessing = MoleculeProcessing(),
                        ) -> dict:
    """
    Creates a new graph dict representation from the given rdkit ``mol`` object using the given 
    MoleculeProcessing ``processing`` instance to do most of the conversion.
    
    :param mol: The rdkit.Mol instance that represents the molecule to be encoded.
    
    :returns: A graph dict
    """
    # --- Domain specific conversion ---
    # Instead of re-inventing the wheel here on how to convert a molecule to a graph dict, we simply use 
    # the already existing MoleculeProcessing class from the chem_mat_data package.
    # This processing class constructs a basic graph dict representation which already contains information 
    # about the node atoms for example as well as and edge_indices property that encodes the bond 
    # connectivity.
    graph = processing.process(
        value=mol,
        double_edges_undirected=False,
    )
    # We'll derive some porperties from the networkx graph representation
    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
    nx_graph = nx.from_numpy_array(adjacency_matrix)
    
    # --- Calculating Atomic Numbers ---
    # We need the atomic numbers of all the atoms in one separate array for the encoding of the molecular
    # graphs later on.
    node_atoms: np.ndarray = np.zeros(shape=graph['node_indices'].shape)
    for i, atom in enumerate(mol.GetAtoms()):
        node_atoms[i] = atom.GetAtomicNum()
    
    graph['node_atoms'] = node_atoms
    
    # --- Connectivity from mol ---
    edges = list(nx_graph.edges())
    edge_indices = np.array(edges, dtype=int)
    # create bidirectional edges by appending flipped (j,i) pairs
    edge_indices = np.concatenate([edge_indices, edge_indices[:, ::-1]], axis=0)
        
    graph['edge_indices'] = edge_indices
    
    # --- Calculating node degree (from networkx graph) ---
    # Use the previously constructed networkx graph to derive node degrees which ensures consistency
    # with any preprocessing that might have modified the graph structure.
    node_degrees: np.ndarray = np.array(
        [nx_graph.degree[int(node_index)] for node_index in graph['node_indices']],
        dtype=float
    )
    graph['node_degrees'] = node_degrees
    
    # --- Calculating node valence ---
    # We also need the information about the valence of the atoms (number of implicitly attached hydrogens)
    # which we get from the mol object in this case.
    node_valences: np.ndarray = np.zeros(shape=graph['node_indices'].shape)
    for i, atom in enumerate(mol.GetAtoms()):
        node_valences[i] = atom.GetNumImplicitHs()
    
    graph['node_valences'] = node_valences
    
    # --- Determining graph properties ---
    # We also add some global graph properties that might be useful for the encoding of the molecular 
    # graphs later on. This includes something like the size of the graph (number of nodes) or the 
    # diameter of the graph
    
    graph_size = nx.number_of_nodes(nx_graph)
    graph['graph_size'] = graph_size
    
    graph['graph_diameter'] = nx.diameter(nx_graph)
    
    return graph


def make_molecule_node_encoder_map(
    dim: int, 
    atoms: List[str] = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si', 'Ge', 'Be', 'Sn', 'B', 'As', 'Se', 'Na', 'Mg', 'Ca', 'Fe', 'Al', 'Cu', 'Zn', 'K', 'Zr', 'Hg'],
    #atoms: List[str] = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
    #atoms: List[str] = [1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 13.0, 14.0, 15.0, 16.0, 17.0, 20.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 33.0, 34.0, 35.0, 38.0, 39.0, 40.0, 41.0, 42.0, 48.0, 50.0, 51.0, 52.0, 53.0, 58.0, 74.0, 80.0, 82.0, 83.0],
    seed: int = None,
) -> dict:
    """
    This function returns a dictionary that will act as a "node_encoder_map" that can be supplied to a HyperNet encoder
    to encode the node properties specifically for a molecular graph (as returned by the "graph_dict_from_mol" function).
    The created encoders will create hypervector encodings of the given dimensionality ``dim``.
    
    This encoder map will contain three key-value pairs:
    - node_atoms: An AtomEncoder instance which encodes the given ``atoms`` list of atom symbols.
    - node_degrees: A CategoricalIntegerEncoder instance which encodes the integer degree of the nodes
    - node_valences: A CategoricalIntegerEncoder instance which encodes the integer valence (number of implicitly 
      attached hydrogens) of the nodes.
      
    :param dim: The dimensionality of the hyperdimensional vectors to be used for encoding.
    :param atoms: A list of atom symbols that should be encoded.
    
    :returns: A dictionary mapping node attribute names to their respective implementations of the AbstractEncoder
        interface.
    """
    return {
        'node_atoms': AtomEncoder(dim=dim, atoms=atoms, seed=seed),
        'node_degrees': CategoricalIntegerEncoder(dim=dim, num_categories=8, seed=seed+10),
        'node_valences': CategoricalIntegerEncoder(dim=dim, num_categories=6, seed=seed+20),
        # 'node_degrees': ContinuousEncoder(dim=dim, size=10.0, bandwidth=2.0, seed=seed+10),
        # 'node_valences': ContinuousEncoder(dim=dim, size=10.0, bandwidth=2.0, seed=seed+20),
    }
    

def make_molecule_node_encoder_map_cont(
    dim: int,
    atoms: List[Union[str, int]] = [1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 13.0, 14.0, 15.0, 16.0, 17.0, 20.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 33.0, 34.0, 35.0, 38.0, 39.0, 40.0, 41.0, 42.0, 48.0, 50.0, 51.0, 52.0, 53.0, 58.0, 74.0, 80.0, 82.0, 83.0],
    seed: int = 0,
) -> Dict[str, AbstractEncoder]:
    return {
        'node_atoms': AtomEncoder(
            dim=dim,
            atoms=atoms,
            seed=seed+10,
        ), 
        'node_degrees': ContinuousEncoder(
            dim=dim,
            size=10.0,
            bandwidth=2.0,
            seed=seed+20,
        ),
        'node_valences': ContinuousEncoder(
            dim=dim,
            size=10.0,
            bandwidth=2.0,
            seed=seed+30,
        ),
    }
    

def make_molecule_graph_encoder_map_cont(
    dim: int,
    max_graph_size: float = 130.0,
    max_graph_diameter: float = 20.0,
    seed: int = None,
) -> Dict[str, AbstractEncoder]:
    """
    
    """
    return {
        'graph_size': ContinuousEncoder(
            dim=dim, 
            size=max_graph_size,
            bandwidth=max(3.0, max_graph_size / 7.0), 
            seed=seed
        ),
        'graph_diameter': ContinuousEncoder(
            dim=dim,
            size=max_graph_diameter,
            bandwidth=max(2.0, max_graph_diameter / 5.0),
            seed=seed+10
        ),
    }

    
def mol_from_graph_dict(graph: dict) -> Chem.Mol:
    
    mol = Chem.RWMol()
    atom_idx_map = {}

    # Add atoms
    for index in graph['node_indices']:
        
        atomic_number: int = int(graph['node_atoms'][index])
        atom = Chem.Atom(atomic_number)
    
        idx = mol.AddAtom(atom)
        atom_idx_map[int(index)] = idx

    # Add bonds
    for i, j in graph['edge_indices']:
        
        valence_i = 8 - pt.GetNOuterElecs(int(graph['node_atoms'][int(i)]))
        valence_j = 8 - pt.GetNOuterElecs(int(graph['node_atoms'][int(j)]))
        # print()
        # print(graph['node_atoms'][int(i)], valence_i)
        # print(graph['node_atoms'][int(j)], valence_j)
        
        num_hs_i = graph['node_valences'][int(i)]
        num_hs_j = graph['node_valences'][int(j)]
        
        num_bonds_i = sum(int(i in edge and j not in edge) for edge in graph['edge_indices'])
        num_bonds_j = sum(int(j in edge and i not in edge) for edge in graph['edge_indices'])
        
        bond_order = math.floor(((valence_i + valence_j) - (num_bonds_i + num_bonds_j) - (num_hs_i + num_hs_j)) / 2)
        
        # Add bond with correct bond order if possible
        if bond_order == 3:
            mol.AddBond(atom_idx_map[int(i)], atom_idx_map[int(j)], Chem.BondType.TRIPLE)
        elif bond_order == 2:
            mol.AddBond(atom_idx_map[int(i)], atom_idx_map[int(j)], Chem.BondType.DOUBLE)
        else:
            mol.AddBond(atom_idx_map[int(i)], atom_idx_map[int(j)], Chem.BondType.SINGLE)

    # Sanitize molecule to update valence and bond orders
    #Chem.SanitizeMol(mol)

    return mol