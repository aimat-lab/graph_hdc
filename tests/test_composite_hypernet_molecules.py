"""
Molecular tests for CompositeHyperNet.

This module tests CompositeHyperNet with real molecular graphs to demonstrate
real-world usage and verify it works with the molecular encoding utilities.
"""

import torch
import pytest
from rdkit import Chem

from graph_hdc.models import CompositeHyperNet
from graph_hdc.special.molecules import make_molecule_node_encoder_map, graph_dict_from_mol


class TestCompositeHyperNetMolecules:
    """Test CompositeHyperNet with molecular graphs."""

    def test_encode_simple_molecule(self):
        """Test encoding a simple molecule (ethanol)."""
        hidden_dim = 1000
        smiles = 'CCO'  # Ethanol

        # Create molecular encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)

        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
            bidirectional=True,
            seed=42
        )

        # Convert SMILES to graph dict
        mol = Chem.MolFromSmiles(smiles)
        graph = graph_dict_from_mol(mol)

        # Encode molecule
        results = encoder.forward_graphs([graph])
        result = results[0]

        # Verify composite structure
        assert 'graph_embedding' in result
        assert 'graph_hv_stack' in result
        assert result['graph_embedding'].shape[-1] == 3 * hidden_dim
        assert result['graph_hv_stack'].shape[-1] == 3 * hidden_dim

        print(f"Encoded {smiles} with {len(graph['node_indices'])} atoms")
        print(f"Embedding shape: {result['graph_embedding'].shape}")

    def test_decode_molecule_nodes(self):
        """Test decoding node information from a molecule."""
        hidden_dim = 2000
        smiles = 'CCO'  # Ethanol: 3 atoms (2 carbons, 1 oxygen)

        # Create molecular encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)

        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
            bidirectional=True,
            seed=42
        )

        # Convert and encode
        mol = Chem.MolFromSmiles(smiles)
        graph = graph_dict_from_mol(mol)

        results = encoder.forward_graphs([graph])
        result = results[0]

        # Decode nodes
        embedding = torch.tensor(result['graph_hv_stack'])
        node_constraints = encoder.decode_order_zero(embedding)

        # Should detect some atoms
        assert len(node_constraints) > 0

        # Count total atoms
        total_atoms = sum(c['num'] for c in node_constraints)
        expected_atoms = len(graph['node_indices'])

        print(f"Expected {expected_atoms} atoms, decoded {total_atoms}")
        print(f"Node constraints: {node_constraints}")

        # Should be close (perfect accuracy not guaranteed with small hidden_dim)
        assert total_atoms >= expected_atoms * 0.7  # At least 70% accuracy

    def test_multiple_molecules_batch(self):
        """Test encoding multiple molecules in batch."""
        hidden_dim = 500
        smiles_list = ['CCO', 'CC', 'CCC']  # Ethanol, Ethane, Propane

        # Create molecular encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)

        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
            bidirectional=True,
            seed=42
        )

        # Convert all molecules
        graphs = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            graph = graph_dict_from_mol(mol)
            graphs.append(graph)

        # Encode all at once
        results = encoder.forward_graphs(graphs, batch_size=3)

        # Should have one result per molecule
        assert len(results) == len(smiles_list)

        # Each result should have composite structure
        for i, result in enumerate(results):
            assert 'graph_embedding' in result
            assert result['graph_embedding'].shape[-1] == 3 * hidden_dim
            print(f"Encoded {smiles_list[i]}: embedding shape {result['graph_embedding'].shape}")

    def test_composite_components_molecular(self):
        """Test that h_0, h_1, g components are meaningful for molecules."""
        hidden_dim = 500
        smiles = 'CC'  # Ethane: 2 carbons, 1 bond

        # Create molecular encoder
        node_encoder_map = make_molecule_node_encoder_map(dim=hidden_dim, seed=42)

        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
            bidirectional=False,  # Test without bidirectional
            seed=42
        )

        # Convert and encode
        mol = Chem.MolFromSmiles(smiles)
        graph = graph_dict_from_mol(mol)

        # Use forward_graph for single molecule
        result = encoder.forward_graph(graph)

        # Extract components
        h_0 = torch.tensor(result['h_0'])
        h_1 = torch.tensor(result['h_1'])
        g = torch.tensor(result['g'])

        # All components should be non-zero
        assert not torch.allclose(h_0, torch.zeros_like(h_0))
        assert not torch.allclose(h_1, torch.zeros_like(h_1))
        assert not torch.allclose(g, torch.zeros_like(g))

        # Verify composite structure
        composite = torch.tensor(result['graph_embedding'])
        reconstructed = torch.cat([h_0, h_1, g], dim=-1)
        assert torch.allclose(composite, reconstructed, atol=1e-5)

        print(f"h_0 norm: {h_0.norm().item():.2f}")
        print(f"h_1 norm: {h_1.norm().item():.2f}")
        print(f"g norm: {g.norm().item():.2f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
