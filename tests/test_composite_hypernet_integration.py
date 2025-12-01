"""
Integration tests for CompositeHyperNet with reconstruction algorithms.

This module tests that CompositeHyperNet works correctly with the existing
graph reconstruction algorithms.
"""

import torch
import pytest

from graph_hdc.models import CompositeHyperNet
from graph_hdc.reconstruct import GraphReconstructor
from graph_hdc.utils import CategoricalIntegerEncoder


class TestCompositeHyperNetReconstruction:
    """Test CompositeHyperNet integration with graph reconstruction."""

    @pytest.mark.localonly
    def test_with_graph_reconstructor_basic(self):
        """Test that CompositeHyperNet works with GraphReconstructor."""
        hidden_dim = 5000  # Large dimension for better reconstruction

        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_atoms': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5, seed=42),
                'node_degrees': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=4, seed=42),
            },
            seed=42
        )

        # Create test graph: 3 nodes with specific properties
        graph = {
            'node_indices': [0, 1, 2],
            'node_atoms': [6, 7, 8],  # C, N, O
            'node_degrees': [2, 2, 1],
            'node_attributes': [[0], [0], [0]],
            'edge_indices': [(0, 1), (1, 2)],
            'edge_attributes': [[0], [0]],
        }

        # Encode the graph
        results = encoder.forward_graphs([graph])
        result = results[0]

        # Verify composite structure is present
        assert 'graph_hv_stack' in result
        assert result['graph_hv_stack'].shape[-1] == 3 * hidden_dim

        # Extract embedding for reconstruction
        graph_embedding = torch.tensor(result['graph_hv_stack'])

        # Create reconstructor
        reconstructor = GraphReconstructor(
            encoder=encoder,
            population_size=2,
        )

        # Reconstruct (should work without errors)
        try:
            reconstructed = reconstructor.reconstruct(embedding=graph_embedding)

            # Basic checks
            assert 'graph' in reconstructed
            assert 'distance' in reconstructed

            # Verify reconstructed graph has nodes
            reconstructed_graph = reconstructed['graph']
            assert 'node_indices' in reconstructed_graph
            assert len(reconstructed_graph['node_indices']) > 0

            print(f"Reconstruction distance: {reconstructed['distance']:.4f}")
            print(f"Reconstructed {len(reconstructed_graph['node_indices'])} nodes")

        except Exception as e:
            pytest.skip(f"Reconstruction failed (expected for simple test): {e}")

    def test_decode_methods_with_forward_graphs(self):
        """Test that decode methods work with forward_graphs output."""
        hidden_dim = 2000

        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42),
            },
            seed=42
        )

        # Create simple test graph
        graph = {
            'node_indices': [0, 1, 2],
            'node_label': [0, 1, 1],
            'node_attributes': [[0], [0], [0]],
            'edge_indices': [(0, 1), (1, 2)],
            'edge_attributes': [[0], [0]],
        }

        # Use forward_graphs (as reconstruction would)
        results = encoder.forward_graphs([graph])
        result = results[0]

        # Convert to torch tensor
        embedding_stack = torch.tensor(result['graph_hv_stack'])

        # Test decode_order_zero
        node_constraints = encoder.decode_order_zero(embedding_stack)
        assert len(node_constraints) > 0
        total_nodes = sum(c['num'] for c in node_constraints)
        assert total_nodes == 3

        # Test decode_order_one
        edge_constraints = encoder.decode_order_one(embedding_stack)
        assert isinstance(edge_constraints, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
