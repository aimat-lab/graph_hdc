from typing import List, Any, Union

import math
import torch
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import rdkit.Chem as Chem
from rdkit.Chem import GetPeriodicTable
from chem_mat_data.processing import MoleculeProcessing
from graph_hdc.utils import AbstractEncoder
from graph_hdc.utils import CategoricalIntegerEncoder

pt = Chem.GetPeriodicTable()


class AtomEncoder(AbstractEncoder):
    
    periodic_table = GetPeriodicTable()
    
    def __init__(self,
                 dim: int,
                 atoms: List[Union[str, int]],
                 seed: int = None,
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
        
        random = np.random.default_rng(seed)
        self.embeddings: torch.Tensor = torch.tensor(random.normal(
            # This scaling is important to have normalized base vectors
            loc=0.0,
            scale=(1.0 / np.sqrt(self.dim)), 
            size=(self.num_categories, self.dim)
        ).astype(np.float32))
        
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



def graph_dict_from_mol(mol: Chem.Mol,
                        processing: MoleculeProcessing = MoleculeProcessing(),
                        ) -> dict:
    """
    Creates a new graph dict representation from the given rdkit ``mol`` object using the given 
    MoleculeProcessing ``processing`` instance to do most of the conversion.
    
    :param mol: The rdkit.Mol instance that represents the molecule to be encoded.
    
    :returns: A graph dict
    """
    # ~ Domain specific conversion
    # Instead of re-inventing the wheel here on how to convert a molecule to a graph dict, we simply use 
    # the already existing MoleculeProcessing class from the chem_mat_data package.
    # This processing class constructs a basic graph dict representation which already contains information 
    # about the node atoms for example as well as and edge_indices property that encodes the bond 
    # connectivity.
    graph = processing.process(
        value=mol,
        double_edges_undirected=False,
    )
    
    # ~ Calculating Atomic Numbers
    # We need the atomic numbers of all the atoms in one separate array for the encoding of the molecular
    # graphs later on.
    node_atoms: np.ndarray = np.zeros(shape=graph['node_indices'].shape)
    for i, atom in enumerate(mol.GetAtoms()):
        node_atoms[i] = atom.GetAtomicNum()
    
    graph['node_atoms'] = node_atoms
    
    # ~ Calculating node degree
    # We also need the information about the node degree later on for the encoding of the molecular graphs
    # this information is not directly included through the processing instance and therefore needs to be 
    # added here using the connectivity information in the edge_indices list.
    node_degrees: np.ndarray = np.zeros(shape=graph['node_indices'].shape)
    for i, j in graph['edge_indices']:
        node_degrees[i] += 1
        node_degrees[j] += 1
        
    graph['node_degrees'] = node_degrees
    
    # ~ Calculating node valence
    # We also need the information about the valence of the atoms (number of implicitly attached hydrogens)
    # which we get from the mol object in this case.
    node_valences: np.ndarray = np.zeros(shape=graph['node_indices'].shape)
    for i, atom in enumerate(mol.GetAtoms()):
        node_valences[i] = atom.GetNumImplicitHs()
    
    graph['node_valences'] = node_valences
    
    return graph


def make_molecule_node_encoder_map(dim: int, 
                                   atoms: List[str] = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si', 'Ge', 'Be', 'Sn', 'B', 'As', 'Se', 'Na', 'Mg', 'Ca', 'Fe', 'Al', 'Cu', 'Zn', 'K', 'Zr', 'Hg'],
                                   #atoms: List[str] = ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']
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
        'node_atoms': AtomEncoder(dim=dim, atoms=atoms),
        'node_degrees': CategoricalIntegerEncoder(dim=dim, num_categories=10),
        'node_valences': CategoricalIntegerEncoder(dim=dim, num_categories=6),
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