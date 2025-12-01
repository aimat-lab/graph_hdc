import os

import torch
from rich.pretty import pprint
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt

from graph_hdc.models import HyperNet, CompositeHyperNet
from graph_hdc.models import HyperNetEnsemble
from graph_hdc.reconstruct import GraphReconstructor, GraphReconstructorAStar
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
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
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


class TestGraphReconstructorAStar:
    """Test the A* search-based graph reconstruction algorithm."""

    def test_reconstruct_molecule(self):
        """Test that A* reconstruction can reconstruct a simple molecule."""

        hidden_dim = 7_000
        #smiles = 'C1=C(F)C(F)=C(CN)C(F)=C1CC(=O)O'  # Ethanol - simple test case
        #smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
        smiles = 'O[S](=O)(=O)C1=CC2C(CC3C2N=C(Cl)C(=C3Cl)Cl)C=C1'
        
        # Create encoder
        node_encoder_map1 = make_molecule_node_encoder_map(dim=hidden_dim, seed=41)
        hyper_net1 = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map1,
            bidirectional=True,
        )
        
        node_encoder_map2 = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
        hyper_net2 = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=4,
            node_encoder_map=node_encoder_map2,
            bidirectional=True,
        )
        
        node_encoder_map3 = make_molecule_node_encoder_map(dim=hidden_dim, seed=43)
        hyper_net3 = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=5,
            node_encoder_map=node_encoder_map3,
            bidirectional=True,
        )
        
        node_encoder_map4 = make_molecule_node_encoder_map(dim=hidden_dim, seed=44)
        hyper_net4 = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=4,
            node_encoder_map=node_encoder_map4,
            bidirectional=True,
        )
        
        node_encoder_map5 = make_molecule_node_encoder_map(dim=hidden_dim, seed=45)
        hyper_net5 = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=6,
            node_encoder_map=node_encoder_map5,
            bidirectional=True,
        )
        
        hyper_net = HyperNetEnsemble(
            [hyper_net1, hyper_net2, hyper_net3, hyper_net4, hyper_net5],
        )

        # Encode molecule
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        results: list[dict] = hyper_net.forward_graphs([graph])
        result: dict = results[0]

        # Create A* reconstructor
        reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            encoder_sim=hyper_net,
            memory_budget=4000,
            time_budget=120.0,
            batch_size=200,
        )

        # Reconstruct
        graph_embedding = torch.tensor(result['graph_embedding'])
        result = reconstructor.reconstruct(embedding=graph_embedding)
        #reconstructor._print_decoding_summary()

        # Verify reconstruction
        assert 'graph' in result
        assert 'embedding' in result
        assert 'distance' in result

        print(f"A* Reconstruction distance: {result['distance']:.4f}")
        print(f"Nodes expanded: {reconstructor.nodes_expanded}")
        print(f"Tree size: {reconstructor.tree_size}")
        print(f"Reconstructed graph nodes: {len(result['graph'].get('node_details', []))}")
        print(f"Reconstructed graph edges: {len(result['graph'].get('edge_indices', []))}")

        # Check reconstructed graph has correct structure
        reconstructed_graph = result['graph']

        # Visualize original and reconstructed molecules side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'A* Reconstruction - Distance: {result["distance"]:.4f}, Nodes Expanded: {reconstructor.nodes_expanded}')

        # Plot original molecule from SMILES (left column)
        mol_original = Chem.MolFromSmiles(smiles)
        img_original = Draw.MolToImage(mol_original, size=(300, 300))
        axes[0].imshow(img_original)
        axes[0].set_title("Original Molecule")
        axes[0].axis('off')

        # Plot reconstructed molecule (right column)
        mol_reconstructed = mol_from_graph_dict(reconstructed_graph)
        img_reconstructed = Draw.MolToImage(mol_reconstructed, size=(300, 300))
        axes[1].imshow(img_reconstructed)
        axes[1].set_title(f"Reconstructed Molecule\n({len(reconstructed_graph['node_details'])} nodes, {len(reconstructed_graph['edge_indices'])} edges)")
        axes[1].axis('off')

        # Save figure to test artifacts
        fig_path = os.path.join(ARTIFACTS_PATH, 'astar_reconstructed_molecule.png')
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"\nVisualization saved to: {fig_path}")
        
    def test_basically_works(self):
        """Test that A* reconstruction can reconstruct a simple molecule."""

        hidden_dim = 2_000
        smiles = 'CCCO'  # Ethanol - simple test case

        # Create encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
        hyper_net = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
            bidirectional=True,
        )

        # Encode molecule
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        results: list[dict] = hyper_net.forward_graphs([graph])
        result: dict = results[0]

        # Create A* reconstructor
        reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            memory_budget=500,
            time_budget=30.0,
            batch_size=50,
        )

        # Reconstruct
        graph_embedding = torch.tensor(result['graph_embedding'])
        result = reconstructor.reconstruct(embedding=graph_embedding)

        # Verify reconstruction
        assert 'graph' in result
        assert 'embedding' in result
        assert 'distance' in result

        print(f"A* Reconstruction distance: {result['distance']:.4f}")
        print(f"Nodes expanded: {reconstructor.nodes_expanded}")
        print(f"Tree size: {reconstructor.tree_size}")
        print(f"Reconstructed graph nodes: {len(result['graph'].get('node_details', []))}")
        print(f"Reconstructed graph edges: {len(result['graph'].get('edge_indices', []))}")

        # Check reconstructed graph has correct structure
        reconstructed_graph = result['graph']

        # Visualize original and reconstructed molecules side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'A* Reconstruction - Distance: {result["distance"]:.4f}, Nodes Expanded: {reconstructor.nodes_expanded}')

        # Plot original molecule from SMILES (left column)
        mol_original = Chem.MolFromSmiles(smiles)
        img_original = Draw.MolToImage(mol_original, size=(300, 300))
        axes[0].imshow(img_original)
        axes[0].set_title("Original Molecule")
        axes[0].axis('off')

        # Plot reconstructed molecule (right column)
        mol_reconstructed = mol_from_graph_dict(reconstructed_graph)
        img_reconstructed = Draw.MolToImage(mol_reconstructed, size=(300, 300))
        axes[1].imshow(img_reconstructed)
        axes[1].set_title(f"Reconstructed Molecule\n({len(reconstructed_graph['node_details'])} nodes, {len(reconstructed_graph['edge_indices'])} edges)")
        axes[1].axis('off')

        # Save figure to test artifacts
        fig_path = os.path.join(ARTIFACTS_PATH, 'astar_reconstructed_molecule.png')
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"\nVisualization saved to: {fig_path}")

    def test_memory_budget(self):
        """Test that memory budget is respected during reconstruction."""

        hidden_dim = 5_000
        smiles = 'CC(C)O'  # Isopropanol

        # Create encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
        hyper_net = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
        )

        # Encode molecule
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        results: list[dict] = hyper_net.forward_graphs([graph])
        result: dict = results[0]

        # Create A* reconstructor with small memory budget
        reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            memory_budget=20,  # Very small budget
            time_budget=10.0,
            batch_size=10,
        )

        # Reconstruct
        graph_embedding = torch.tensor(result['graph_hv_stack'])
        result = reconstructor.reconstruct(embedding=graph_embedding)

        # Should still return a result despite memory limit
        assert 'graph' in result
        assert 'distance' in result

        # Tree size should not exceed budget (much)
        assert reconstructor.tree_size <= reconstructor.memory_budget * 2

    def test_time_budget(self):
        """Test that time budget is respected during reconstruction."""

        hidden_dim = 5_000
        smiles = 'CCCC'  # Butane

        # Create encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
        hyper_net = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
        )

        # Encode molecule
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        results: list[dict] = hyper_net.forward_graphs([graph])
        result: dict = results[0]

        # Create A* reconstructor with very short time budget
        reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            memory_budget=1000,
            time_budget=2.0,  # Very short time
            batch_size=10,
        )

        # Measure reconstruction time
        import time
        start = time.time()
        graph_embedding = torch.tensor(result['graph_hv_stack'])
        result = reconstructor.reconstruct(embedding=graph_embedding)
        elapsed = time.time() - start

        # Should return within time budget (plus small overhead)
        assert elapsed < reconstructor.time_budget + 1.0

        # Should still return a valid result
        assert 'graph' in result
        assert 'distance' in result

    def test_degree_constraints(self):
        """Test that degree constraints are properly enforced."""

        hidden_dim = 10_000

        # Create a simple graph with explicit degree constraints
        graph = {
            'node_indices': [0, 1, 2],
            'node_details': [
                {'node_atoms': 'C', 'node_degrees': 2},  # Can have max 2 edges
                {'node_atoms': 'N', 'node_degrees': 3},  # Can have max 3 edges
                {'node_atoms': 'O', 'node_degrees': 1},  # Can have max 1 edge
            ],
            'node_attributes': [[6], [7], [8]],
            'edge_indices': [(0, 1), (1, 2)],
            'edge_attributes': [[1], [1]],
            'node_atoms': ['C', 'N', 'O'],
            'node_degrees': [2, 3, 1],
        }

        from graph_hdc.utils import CategoricalIntegerEncoder

        # Create encoder with degree-aware encoding
        node_encoder_map = {
            'node_atoms': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=10),
            'node_degrees': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5),
        }

        hyper_net = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
        )

        # Encode graph
        results: list[dict] = hyper_net.forward_graphs([graph])
        result: dict = results[0]

        # Create A* reconstructor with constraint validation
        reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            memory_budget=500,
            time_budget=10.0,
            validate_constraints=True,  # Enable constraint checking
        )

        # Reconstruct
        graph_embedding = torch.tensor(result['graph_hv_stack'])
        result = reconstructor.reconstruct(embedding=graph_embedding)

        # Verify reconstruction respects degree constraints
        reconstructed = result['graph']

        # Count edges per node
        node_degrees = [0] * len(reconstructed['node_indices'])
        for i, j in reconstructed['edge_indices']:
            node_degrees[i] += 1
            node_degrees[j] += 1

        # Check that no node exceeds its degree constraint
        for idx, degree in enumerate(node_degrees):
            max_degree = reconstructed['node_details'][idx]['node_degrees']
            assert degree <= max_degree, f"Node {idx} has degree {degree} but max is {max_degree}"

        print(f"Degree constraints test passed")
        print(f"Original edges: {graph['edge_indices']}")
        print(f"Reconstructed edges: {reconstructed['edge_indices']}")

    def test_comparison_with_evolutionary(self):
        """Compare A* reconstruction with the evolutionary approach."""

        hidden_dim = 10_000
        smiles = 'CCO'

        # Create encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
        hyper_net = HyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
        )

        # Encode molecule
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        results: list[dict] = hyper_net.forward_graphs([graph])
        result: dict = results[0]
        graph_embedding = torch.tensor(result['graph_hv_stack'])

        # Test evolutionary approach
        evolutionary_reconstructor = GraphReconstructor(
            encoder=hyper_net,
            population_size=3,
        )

        import time

        start = time.time()
        evolutionary_result = evolutionary_reconstructor.reconstruct(embedding=graph_embedding)
        evolutionary_time = time.time() - start

        # Test A* approach
        astar_reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            memory_budget=500,
            time_budget=30.0,
        )

        start = time.time()
        astar_result = astar_reconstructor.reconstruct(embedding=graph_embedding)
        astar_time = time.time() - start

        # Compare results
        print(f"\nComparison Results:")
        print(f"Evolutionary - Distance: {evolutionary_result['distance']:.4f}, Time: {evolutionary_time:.2f}s")
        print(f"A* Search    - Distance: {astar_result['distance']:.4f}, Time: {astar_time:.2f}s")
        print(f"A* nodes expanded: {astar_reconstructor.nodes_expanded}, tree size: {astar_reconstructor.tree_size}")

        # Both should produce reasonable reconstructions
        assert evolutionary_result['distance'] < 1.0
        assert astar_result['distance'] < 1.0

    def test_empty_graph(self):
        """Test reconstruction when no nodes are detected."""

        hidden_dim = 1_000

        # Create encoder
        from graph_hdc.utils import CategoricalIntegerEncoder
        node_encoder_map = {
            'node_atoms': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=10),
        }

        hyper_net = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
        )

        # Create A* reconstructor
        reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            memory_budget=100,
            time_budget=5.0,
        )

        # Try to reconstruct from zero embedding
        zero_embedding = torch.zeros(hidden_dim)
        result = reconstructor.reconstruct(embedding=zero_embedding)

        # Should return empty graph
        assert 'graph' in result
        assert len(result['graph']['node_indices']) == 0
        assert len(result['graph']['edge_indices']) == 0

    def test_deterministic_encoding(self):
        """Test that encoding the same molecule twice gives identical embeddings."""

        hidden_dim = 10_000
        smiles = 'CCO'  # Ethanol

        # Create encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
        hyper_net = HyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
        )

        # Encode the same molecule twice
        graph1: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        graph2: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))

        results1: list[dict] = hyper_net.forward_graphs([graph1])
        results2: list[dict] = hyper_net.forward_graphs([graph2])

        embedding1 = torch.tensor(results1[0]['graph_hv_stack'])
        embedding2 = torch.tensor(results2[0]['graph_hv_stack'])

        # Calculate distance between the two encodings
        from graph_hdc.reconstruct import cosine_distance
        distance = cosine_distance(embedding1, embedding2)

        print(f"\nDeterministic encoding test:")
        print(f"Distance between two encodings of same molecule: {distance:.10f}")

        # Distance should be very close to 0 (allowing for numerical precision)
        assert distance < 1e-6, f"Distance {distance} is too large for identical molecules"

        # Also check that embeddings are exactly equal
        assert torch.allclose(embedding1, embedding2, atol=1e-8), "Embeddings should be identical"

    def test_duplicate_nodes_handling(self):
        """
        Regression test for duplicate node handling bug.

        Tests that when a graph has multiple nodes of the same type (e.g., 3 carbons),
        the reconstruction process doesn't lose track of nodes. This was a bug where
        the initial alphabet computation would remove ALL instances of a node type
        instead of just the one that was used in the initial graph.

        Bug: line 471 was using list comprehension `[n for n in alphabet if n != used_node]`
        which removed all equal nodes instead of just one instance.

        Fix: Use `alphabet.remove(used_node)` which removes only the first occurrence.
        """

        hidden_dim = 10_000
        # Use a molecule with many duplicate atoms to trigger the bug
        # Benzene (C6H6) has 6 carbons and 6 hydrogens - perfect for testing duplicates
        smiles = 'C1=CC=CC=C1'  # Benzene - 6 carbons (duplicate nodes)

        # Create encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)
        hyper_net = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
            bidirectional=True,
        )

        # Encode molecule
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles(smiles))
        original_node_count = len(graph['node_indices'])

        results: list[dict] = hyper_net.forward_graphs([graph])
        result: dict = results[0]

        # Create A* reconstructor
        reconstructor = GraphReconstructorAStar(
            encoder=hyper_net,
            memory_budget=2000,
            time_budget=60.0,
            batch_size=100,
        )

        # Reconstruct - this should not lose any nodes
        graph_embedding = torch.tensor(result['graph_embedding'])
        reconstructed_result = reconstructor.reconstruct(embedding=graph_embedding)

        # Verify reconstruction
        assert 'graph' in reconstructed_result
        reconstructed_graph = reconstructed_result['graph']
        reconstructed_node_count = len(reconstructed_graph['node_details'])

        print(f"\nDuplicate nodes handling test:")
        print(f"Original molecule: {smiles}")
        print(f"Original node count: {original_node_count}")
        print(f"Reconstructed node count: {reconstructed_node_count}")
        print(f"Reconstruction distance: {reconstructed_result['distance']:.4f}")
        print(f"Nodes expanded: {reconstructor.nodes_expanded}")
        print(f"Tree size: {reconstructor.tree_size}")

        # The key assertion: reconstructed graph should have the same number of nodes
        # This would fail with the bug because nodes would disappear from the alphabet
        assert reconstructed_node_count == original_node_count, \
            f"Node count mismatch! Original: {original_node_count}, Reconstructed: {reconstructed_node_count}. " \
            f"This indicates nodes disappeared during reconstruction (the duplicate node bug)."

        # Additional verification: check that the reconstruction is reasonable
        assert reconstructed_result['distance'] < 1.0, \
            f"Reconstruction distance {reconstructed_result['distance']:.4f} is too high"