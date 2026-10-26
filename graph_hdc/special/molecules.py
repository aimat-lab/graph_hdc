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
from rdkit.Chem import rdCIPLabeler
from chem_mat_data.processing import MoleculeProcessing
from graph_hdc.utils import AbstractEncoder
from graph_hdc.utils import CategoricalIntegerEncoder
from graph_hdc.utils import ContinuousEncoder
from graph_hdc.utils import to_unit_modulus

pt = Chem.GetPeriodicTable()


class AtomEncoder(AbstractEncoder):
    """
    Categorical encoder for the chemical element of an atom: one independent random hypervector per element in
    ``atoms`` plus one for unknown elements.

    :param unit_modulus: If True (default), the random element vectors are unitary: their Fourier coefficients
        keep the phases of the Gaussian draw but all have magnitude one, as in FHRR. Binding (circular
        convolution) with such vectors preserves norms and does not concentrate the bound hypervectors on a few
        Fourier components. False keeps the Gaussian vectors of the original implementation. The vectors stay
        nearly orthogonal either way.
    """

    periodic_table = GetPeriodicTable()

    def __init__(self,
                 dim: int,
                 atoms: List[Union[str, int]],
                 seed: int = 0,
                 unit_modulus: bool = True,
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
        self.embeddings = self.dist.sample((self.num_categories, dim)).to(torch.float64)
        self.unit_modulus = unit_modulus
        if unit_modulus:
            # same random draw, Fourier magnitudes set to one (the inverse FFT of a unit-modulus spectrum has norm 1)
            self.embeddings = torch.fft.ifft(to_unit_modulus(torch.fft.fft(self.embeddings, dim=-1)), dim=-1).real
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


# The categories of the optional "node_stereo" atom attribute (see ``stereo_labels``): the CIP descriptors of
# stereocenters (R/S, and r/s for pseudoasymmetric centers such as in cis/trans-1,4-disubstituted cyclohexanes)
# and of stereogenic double bonds (E/Z, assigned to both atoms of the bond). Index 0 marks all other atoms,
# including stereocenters whose configuration is not specified in the input.
STEREO_LABELS: List[str] = ['none', 'R', 'S', 'r', 's', 'E', 'Z']


def stereo_labels(mol: Chem.Mol) -> np.ndarray:
    """
    Returns the index of the CIP stereo descriptor (see ``STEREO_LABELS``) of every atom of the given ``mol``,
    as determined by RDKit's CIP labeler. Only stereo elements that are specified in ``mol`` receive a
    descriptor. Descriptors of other kinds (e.g. M/P for axial chirality) are ignored, and an atom that is both
    a stereocenter and part of a stereogenic double bond keeps the descriptor of the stereocenter.

    The descriptors are determined on a copy of the molecule in canonical atom order and mapped back, because
    the labeler can assign different descriptors to a few double bonds (e.g. next to a deprotonated azole ring)
    depending on the atom order of the input.

    :param mol: The rdkit.Mol instance, which is not modified.

    :returns: A float array with one category index per atom.
    """
    # Atom j of the canonical copy is atom order[j] of mol. If the round trip through the canonical SMILES does
    # not reproduce the atoms (e.g. explicit hydrogen atoms), the labels are determined in the input order.
    copy = Chem.Mol(mol)
    smiles = Chem.MolToSmiles(copy)
    order = [int(i) for i in copy.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']]
    canonical = Chem.MolFromSmiles(smiles)
    if canonical is None or canonical.GetNumAtoms() != mol.GetNumAtoms():
        canonical, order = Chem.Mol(mol), list(range(mol.GetNumAtoms()))

    # The labeler stores its results as "_CIPCode" properties, which the legacy stereo perception of
    # MolFromSmiles also sets, so these properties are removed first.
    for item in list(canonical.GetAtoms()) + list(canonical.GetBonds()):
        item.ClearProp('_CIPCode')
    rdCIPLabeler.AssignCIPLabels(canonical)

    labels = np.zeros(shape=(mol.GetNumAtoms(), ), dtype=float)
    for bond in canonical.GetBonds():
        if bond.HasProp('_CIPCode') and bond.GetProp('_CIPCode') in ('E', 'Z'):
            index = STEREO_LABELS.index(bond.GetProp('_CIPCode'))
            labels[order[bond.GetBeginAtomIdx()]] = index
            labels[order[bond.GetEndAtomIdx()]] = index
    for atom in canonical.GetAtoms():
        if atom.HasProp('_CIPCode') and atom.GetProp('_CIPCode') in ('R', 'S', 'r', 's'):
            labels[order[atom.GetIdx()]] = STEREO_LABELS.index(atom.GetProp('_CIPCode'))

    return labels


class StereoEncoder(AbstractEncoder):
    """
    Encodes the optional "node_stereo" atom attribute (see ``STEREO_LABELS``). Every stereo descriptor gets a
    random hypervector, like the categories of the other atom attributes. The first category (no descriptor) is
    encoded as the identity of the circular convolution (1, 0, ..., 0), so that binding it leaves the node
    hypervector unchanged: molecules without specified stereo elements receive the same fingerprint as without
    the stereo attribute.

    The random hypervectors are drawn from a private generator, so creating this encoder does not change the
    global random state that the other encoders and the downstream models depend on.

    :param unit_modulus: If True (default, as for AtomEncoder), the random descriptor vectors get unit Fourier
        magnitudes with the phases of the Gaussian draw. The identity vector already has a flat spectrum.
    """

    def __init__(self,
                 dim: int,
                 seed: int = 0,
                 unit_modulus: bool = True,
                 ) -> None:
        AbstractEncoder.__init__(self, dim, seed)
        self.num_categories = len(STEREO_LABELS)
        self.unit_modulus = unit_modulus

        generator = torch.Generator().manual_seed(seed)
        self.embeddings = (torch.randn((self.num_categories, dim), generator=generator) / np.sqrt(dim))
        self.embeddings = self.embeddings.to(torch.float64)
        if unit_modulus:
            # same random draw, Fourier magnitudes set to one (the inverse FFT of a unit-modulus spectrum has norm 1)
            self.embeddings = torch.fft.ifft(to_unit_modulus(torch.fft.fft(self.embeddings, dim=-1)), dim=-1).real
        self.embeddings[0] = 0.0
        self.embeddings[0, 0] = 1.0

    def encode(self, value: Any) -> torch.Tensor:
        return self.embeddings[int(value)]

    def decode(self, hv: torch.Tensor) -> Any:
        distances = torch.norm(self.embeddings - hv, dim=1)
        return int(torch.argmin(distances))

    def get_encoder_hv_dict(self) -> Dict[Any, torch.Tensor]:
        return dict(enumerate(self.embeddings))


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
                        hydrogens: str = 'implicit',
                        stereo: bool = False,
                        ) -> dict:
    """
    Creates a new graph dict representation from the given rdkit ``mol`` object using the given
    MoleculeProcessing ``processing`` instance to do most of the conversion.

    :param mol: The rdkit.Mol instance that represents the molecule to be encoded.
    :param hydrogens: How the hydrogen count of each atom ("node_valences") is determined. "implicit" uses
        ``GetNumImplicitHs``, which is 0 for every atom written in brackets ([nH], [NH3+], [C@@H], ...).
        "total" uses ``GetTotalNumHs`` and counts all bonded hydrogens. The default keeps the original
        behavior, on which the decoders rely.
    :param stereo: Whether to add the "node_stereo" attribute with the CIP stereo descriptor of every atom
        (see ``stereo_labels``). The encoder maps only encode it when they are created with ``stereo=True``.

    :returns: A graph dict
    """
    if hydrogens not in ('implicit', 'total'):
        raise ValueError(f'unknown hydrogens mode "{hydrogens}", expected "implicit" or "total"')
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
    # Keep edges directional (not bidirectional)
    # Previously: edge_indices = np.concatenate([edge_indices, edge_indices[:, ::-1]], axis=0)

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
    # We also need the information about the valence of the atoms (number of attached hydrogens)
    # which we get from the mol object in this case.
    node_valences: np.ndarray = np.zeros(shape=graph['node_indices'].shape)
    for i, atom in enumerate(mol.GetAtoms()):
        node_valences[i] = atom.GetTotalNumHs() if hydrogens == 'total' else atom.GetNumImplicitHs()
    
    graph['node_valences'] = node_valences

    # --- Stereochemistry (optional) ---
    if stereo:
        graph['node_stereo'] = stereo_labels(mol)

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
    unit_modulus: bool = False,
    stereo: bool = False,
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
    - node_stereo (only if ``stereo``): A StereoEncoder instance which encodes the CIP stereo descriptors.

    :param dim: The dimensionality of the hyperdimensional vectors to be used for encoding.
    :param atoms: A list of atom symbols that should be encoded.
    :param unit_modulus: Whether the element vectors get unit Fourier magnitudes (see AtomEncoder). Default False:
        this categorical encoder map, which the decoding code uses, keeps the original Gaussian element vectors. The
        unit-modulus default of 2026-10-09 applies to the continuous encoder maps (``make_molecule_*_map_cont``).
    :param stereo: Whether to encode the "node_stereo" attribute, which the graph dicts then have to contain
        (``graph_dict_from_mol(..., stereo=True)``). Its codebook follows ``unit_modulus``.

    :returns: A dictionary mapping node attribute names to their respective implementations of the AbstractEncoder
        interface.
    """
    encoder_map = {
        'node_atoms': AtomEncoder(dim=dim, atoms=atoms, seed=seed, unit_modulus=unit_modulus),
        'node_degrees': CategoricalIntegerEncoder(dim=dim, num_categories=8, seed=seed+10),
        'node_valences': CategoricalIntegerEncoder(dim=dim, num_categories=6, seed=seed+20),
        # 'node_degrees': ContinuousEncoder(dim=dim, size=10.0, bandwidth=2.0, seed=seed+10),
        # 'node_valences': ContinuousEncoder(dim=dim, size=10.0, bandwidth=2.0, seed=seed+20),
    }
    if stereo:
        encoder_map['node_stereo'] = StereoEncoder(dim=dim, seed=seed+40, unit_modulus=unit_modulus)

    return encoder_map


def make_molecule_node_encoder_map_cont(
    dim: int,
    atoms: List[Union[str, int]] = [1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 13.0, 14.0, 15.0, 16.0, 17.0, 20.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 33.0, 34.0, 35.0, 38.0, 39.0, 40.0, 41.0, 42.0, 48.0, 50.0, 51.0, 52.0, 53.0, 58.0, 74.0, 80.0, 82.0, 83.0],
    seed: int = 0,
    unit_modulus: bool = True,
    stereo: bool = False,
) -> Dict[str, AbstractEncoder]:
    """
    Node encoders of HDF: the element (categorical) and the heavy-atom degree and hydrogen count (fractional
    power encoding). ``unit_modulus`` (default True) gives all codebook vectors unit Fourier magnitudes (see
    AtomEncoder and ContinuousEncoder); False reproduces the original encoder. With ``stereo``, the map also
    encodes the "node_stereo" attribute (StereoEncoder, whose codebook follows ``unit_modulus``).
    """
    encoder_map = {
        'node_atoms': AtomEncoder(
            dim=dim,
            atoms=atoms,
            seed=seed+10,
            unit_modulus=unit_modulus,
        ),
        'node_degrees': ContinuousEncoder(
            dim=dim,
            size=10.0,
            bandwidth=2.0,
            seed=seed+20,
            unit_modulus=unit_modulus,
        ),
        'node_valences': ContinuousEncoder(
            dim=dim,
            size=10.0,
            bandwidth=2.0,
            seed=seed+30,
            unit_modulus=unit_modulus,
        ),
    }
    if stereo:
        encoder_map['node_stereo'] = StereoEncoder(dim=dim, seed=seed+40, unit_modulus=unit_modulus)

    return encoder_map


def make_molecule_graph_encoder_map_cont(
    dim: int,
    max_graph_size: float = 130.0,
    max_graph_diameter: float = 20.0,
    seed: int = None,
    unit_modulus: bool = True,
) -> Dict[str, AbstractEncoder]:
    """
    Graph-level encoders of HDF: graph size and graph diameter (fractional power encoding). ``unit_modulus``
    (default True) gives the base vectors unit Fourier magnitudes (see ContinuousEncoder); False reproduces the
    original encoder.
    """
    return {
        'graph_size': ContinuousEncoder(
            dim=dim,
            size=max_graph_size,
            bandwidth=max(3.0, max_graph_size / 7.0),
            seed=seed,
            unit_modulus=unit_modulus,
        ),
        'graph_diameter': ContinuousEncoder(
            dim=dim,
            size=max_graph_diameter,
            bandwidth=max(2.0, max_graph_diameter / 5.0),
            seed=seed+10,
            unit_modulus=unit_modulus,
        ),
    }


def make_value_graph_encoder(
    values: List[float],
    dim: int,
    seed: int,
    unit_modulus: bool = True,
    min_bandwidth: float = 1.0,
    quantile: float = 0.01,
) -> ContinuousEncoder:
    """
    Fractional power encoder for a further continuous graph-level attribute, for example Crippen's logP (ex_26). It is
    built like the graph size encoder of ``make_molecule_graph_encoder_map_cont``, whose bandwidth is one seventh of
    the value range: here the range between the ``quantile`` and 1 - ``quantile`` quantiles of ``values`` (the
    attribute over the whole dataset), so that a few extreme molecules do not coarsen the resolution of the bulk, and
    at least ``min_bandwidth``. With unit-modulus base vectors, the similarity of two encoded values depends only on
    their difference (about zero at one bandwidth), so no offset is needed. Add the encoder to the graph encoder map
    under the key of the graph dict entry that holds the value (e.g. "graph_clogp"); HyperNet then sums its hypervector
    with those of the graph size and diameter.

    :param values: The attribute values of all molecules of the dataset (only their quantiles are used).
    :param dim: The dimension of the hypervectors.
    :param seed: The seed of the random base vector. Use an offset that no other encoder of the map uses.
    :param unit_modulus: Whether the base vector has unit Fourier magnitudes (see ContinuousEncoder). Without it, the
        random magnitudes are raised to the power value / bandwidth, which grows without bound for negative values.
    :param min_bandwidth: The lower bound of the bandwidth, in units of the attribute.
    :param quantile: The lower quantile of the range (0 uses the minimum and the maximum).
    """
    low, high = (float(x) for x in np.quantile(np.asarray(values, dtype=float), [quantile, 1.0 - quantile]))
    return ContinuousEncoder(
        dim=dim,
        size=high - low,
        bandwidth=max(min_bandwidth, (high - low) / 7.0),
        seed=seed,
        unit_modulus=unit_modulus,
    )


def mol_from_graph_dict(graph: dict) -> Chem.Mol:
    """
    Create an RDKit molecule from a graph dict representation.

    Handles bidirectional edges by only adding each unique bond once.

    :param graph: Graph dict with node_atoms, node_valences, edge_indices, etc.
    :returns: RDKit Mol object
    """
    mol = Chem.RWMol()
    atom_idx_map = {}

    # Add atoms
    for index in graph['node_indices']:

        atomic_number: int = int(graph['node_atoms'][index])
        atom = Chem.Atom(atomic_number)

        idx = mol.AddAtom(atom)
        atom_idx_map[int(index)] = idx

    # Add bonds - handle bidirectional edges by tracking added bonds
    added_bonds = set()
    for i, j in graph['edge_indices']:
        # Normalize bond to (min, max) to avoid adding duplicate bonds
        bond_key = tuple(sorted([int(i), int(j)]))

        if bond_key in added_bonds:
            continue  # Skip if already added

        added_bonds.add(bond_key)
        i, j = int(i), int(j)

        valence_i = 8 - pt.GetNOuterElecs(int(graph['node_atoms'][i]))
        valence_j = 8 - pt.GetNOuterElecs(int(graph['node_atoms'][j]))

        num_hs_i = graph['node_valences'][i]
        num_hs_j = graph['node_valences'][j]

        # Count unique bonds (not directional edges)
        unique_bonds_i = len(set(tuple(sorted([i, other])) for other in range(len(graph['node_indices'])) if (i, other) in graph['edge_indices'] or (other, i) in graph['edge_indices']))
        unique_bonds_j = len(set(tuple(sorted([j, other])) for other in range(len(graph['node_indices'])) if (j, other) in graph['edge_indices'] or (other, j) in graph['edge_indices']))

        num_bonds_i = unique_bonds_i - 1  # -1 because we're currently adding one
        num_bonds_j = unique_bonds_j - 1

        bond_order = math.floor(((valence_i + valence_j) - (num_bonds_i + num_bonds_j) - (num_hs_i + num_hs_j)) / 2)
        bond_order = max(1, bond_order)  # Ensure at least single bond

        # Add bond with correct bond order
        if bond_order == 3:
            mol.AddBond(atom_idx_map[i], atom_idx_map[j], Chem.BondType.TRIPLE)
        elif bond_order == 2:
            mol.AddBond(atom_idx_map[i], atom_idx_map[j], Chem.BondType.DOUBLE)
        else:
            mol.AddBond(atom_idx_map[i], atom_idx_map[j], Chem.BondType.SINGLE)

    # Sanitize molecule to update valence and bond orders
    #Chem.SanitizeMol(mol)

    return mol