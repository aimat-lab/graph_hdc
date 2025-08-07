import os

import torch
from rich.pretty import pprint
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt

from graph_hdc.models import HyperNet
from graph_hdc.reconstruct import GraphReconstructor
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.special.molecules import make_molecule_node_encoder_map
from graph_hdc.special.molecules import mol_from_graph_dict

from .utils import ARTIFACTS_PATH


class TestGraphReconstructor:
    
    def test_basically_works(self):
        
        hidden_dim = 25_000
        #smiles = 'C(F)(F)(F)C(C(O)=O)CC(CN)C=CO'
        #smiles = 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'
        #smiles = 'CC(O)C1=CC=CC=C1'
        smiles = 'C1=CC=CC=C1COCC2=CC=CC=C2'
        #smiles = 'CCO'
        #smiles = 'C1=C(Cl)C=C(Cl)C=C1CN(C)C'
        #smiles = 'CN(CC1=CC=CC=C1)C(=O)C2=CC3=CC(=CC=C3S2)OC'
        
        ## --- molecule encoder ---
        # This is how we encoder molecules into hypervectors.
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim)
        hyper_net = HyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
            #device='cuda',
        )
        
        # We can construct a graph dict from a SMILES string using the special 
        # graph_dict_from_mol function
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        
        # Finally, the encoder net supports the direct conversion of a graph dict to a hyper vector with 
        # the forward_graphs method.
        results: list[dict] = hyper_net.forward_graphs([graph]) 
        result: dict = results[0]
        pprint(result)
        pprint(result['graph_hv_stack'].shape)
        
        ## --- graph reconstructor ---
        # The graph reconstructor needs to get the same encoder object as an argument 
        
        reconstructor = GraphReconstructor(
            encoder=hyper_net,
            population_size=3,
        )
        
        #graph_embedding = torch.tensor(result['graph_embedding'])
        graph_embedding = torch.tensor(result['graph_hv_stack'])
        result: dict = reconstructor.reconstruct(
            embedding=graph_embedding,
        )
    
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'DIM: {hidden_dim}')

        # Plot the molecule from SMILES using RDKit (first column)
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        axes[0].imshow(img)
        axes[0].set_title("Molecule from SMILES")
        axes[0].axis('off')

        # Plot the reconstructed graph structure (second column)
        graph_dict = result["graph"]
        G = nx.Graph()
        for idx, (atom, nhs, deg) in enumerate(zip(graph_dict["node_atoms"], graph_dict['node_valences'], graph_dict['node_degrees'])):
            G.add_node(idx, label=f'{atom},{nhs} ({deg})')
        for edge in graph_dict["edge_indices"]:
            G.add_edge(edge[0], edge[1])

        pos = nx.spring_layout(G, seed=42)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, ax=axes[1], with_labels=True, labels=labels, node_color='lightblue', node_size=500)
        axes[1].set_title(f"Reconstructed Graph - Distance: {result['distance']:.2f}")
        axes[1].axis('off')

        # Plot the molecule from SMILES again (third column)
        mol_decoded = mol_from_graph_dict(graph_dict)
        
        img2 = Draw.MolToImage(mol_decoded, size=(300, 300))
        axes[2].imshow(img2)
        axes[2].set_title("Molecule from SMILES (again)")
        axes[2].axis('off')

        fig_path = os.path.join(ARTIFACTS_PATH, 'reconstructed_molecule.png')
        fig.savefig(fig_path, bbox_inches='tight')