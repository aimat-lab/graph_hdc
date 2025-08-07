import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rich.pretty import pprint

from graph_hdc.special.molecules import AbstractEncoder
from graph_hdc.special.molecules import make_molecule_node_encoder_map
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.models import HyperNet
from rdkit.Chem import Draw

# The size of the final embedding
HIDDEN_DIM: int = 2048

# The number of message passing steps
DEPTH: int = 2

# The smiles to be encoded
SMILES:str = 'CC(=O)OC1=CC=CC=C1C(=O)O'

# --- 1. Constructing the Encoder ---

# The hyper net encoder needs to know how to encode the individual nodes of a 
# molecular graph. This is done by providing a "node encoder map" which contains 
# the different attributes of a node (such as atom type, degree etc.) and 
# define an encoder object which will handle those.
# the `make_molecule_node_encoder_map` function creates such a map for the
# given hidden dimension.
node_encoder_map: dict[str, AbstractEncoder] = make_molecule_node_encoder_map(
    dim=HIDDEN_DIM,
)

encoder = HyperNet(
    hidden_dim=HIDDEN_DIM,
    depth=DEPTH,
    node_encoder_map=node_encoder_map,
)

# --- 2. Encoding a Sample Molecule ---

# The SMILES string has to be converted to a graph dictionary representation.
# This is first done by converting it into a Mol object and then using the 
# function `graph_dict_from_mol`.
mol: Chem.Mol = Chem.MolFromSmiles(SMILES)
graph: dict = graph_dict_from_mol(mol)

# `forward_graphs` takes a list of graph dictionary representations of molecules as 
# the argument and returns a list of dictionaries containing the results of the 
# forward pass through the encoder.
results: list = encoder.forward_graphs([graph])
result: dict[str, np.ndarray] = results[0]

print('--- result ---')
pprint(result)
pprint(result['graph_embedding'])

# Plot the molecule and show it
img = Draw.MolToImage(mol, size=(300, 300))
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis('off')
plt.title('Molecule Structure')
plt.show()