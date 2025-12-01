"""
Tests for CompositeHyperNet class.

This module contains comprehensive tests for the CompositeHyperNet implementation,
including tests for encoding, decoding, and edge cases.
"""

import torch
import numpy as np
import pytest
from torch_geometric.data import Data

from graph_hdc.models import CompositeHyperNet, HyperNet
from graph_hdc.utils import CategoricalIntegerEncoder


class TestCompositeHyperNetBasics:
    """Test basic functionality of CompositeHyperNet."""

    def test_initialization(self):
        """Test that CompositeHyperNet initializes correctly and inherits from HyperNet."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            }
        )

        # Verify inheritance
        assert isinstance(encoder, HyperNet)
        assert isinstance(encoder, CompositeHyperNet)

        # Verify attributes inherited correctly
        assert encoder.hidden_dim == hidden_dim
        assert encoder.depth == 3
        assert len(encoder.node_encoder_map) == 1

    def test_forward_output_shapes(self):
        """Test that forward() produces correct output shapes."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            }
        )

        # Create simple graph: 3 nodes, 2 edges
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        result = encoder.forward(data)

        # Check all expected keys are present
        assert 'graph_embedding' in result
        assert 'graph_hv_stack' in result
        assert 'h_0' in result
        assert 'h_1' in result
        assert 'g' in result

        # Check shapes
        assert result['graph_embedding'].shape == (1, 3 * hidden_dim)
        assert result['graph_hv_stack'].shape == (1, encoder.depth + 1, 3 * hidden_dim)
        assert result['h_0'].shape == (1, hidden_dim)
        assert result['h_1'].shape == (1, hidden_dim)
        assert result['g'].shape == (1, hidden_dim)

    def test_composite_embedding_structure(self):
        """Test that composite embedding is correctly concatenated from h_0, h_1, g."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Create simple graph
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = encoder.forward(data)

        # Verify that graph_embedding = h_0 | h_1 | g
        composite = result['graph_embedding']
        h_0 = result['h_0']
        h_1 = result['h_1']
        g = result['g']

        reconstructed = torch.cat([h_0, h_1, g], dim=-1)
        assert torch.allclose(composite, reconstructed, atol=1e-6)

    def test_batch_processing(self):
        """Test that CompositeHyperNet correctly handles batched graphs."""
        hidden_dim = 50
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Create batch with 2 graphs
        # Graph 0: 2 nodes, 1 edge
        # Graph 1: 3 nodes, 2 edges
        data = Data(
            x=torch.zeros(5, hidden_dim),
            edge_index=torch.tensor([[0, 2, 3], [1, 3, 4]], dtype=torch.long),
            batch=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            node_label=torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)
        )

        result = encoder.forward(data)

        # Check batch dimensions
        assert result['graph_embedding'].shape == (2, 3 * hidden_dim)
        assert result['h_0'].shape == (2, hidden_dim)
        assert result['h_1'].shape == (2, hidden_dim)
        assert result['g'].shape == (2, hidden_dim)


class TestCompositeHyperNetDecoding:
    """Test decoding methods of CompositeHyperNet."""

    def test_decode_order_zero_basic(self):
        """Test basic node decoding from composite embedding."""
        hidden_dim = 1000
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5, seed=42)
            },
            seed=42
        )

        # Create graph with known nodes: 2 label-0, 1 label-1, 1 label-2
        data = Data(
            x=torch.zeros(4, hidden_dim),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 0, 1, 2], dtype=torch.long)
        )

        result = encoder.forward(data)
        embedding = result['graph_embedding']

        # Decode nodes
        node_constraints = encoder.decode_order_zero(embedding)

        # Verify correct number of node types detected
        assert len(node_constraints) > 0

        # Count total nodes
        total_nodes = sum(c['num'] for c in node_constraints)
        assert total_nodes == 4

    def test_decode_order_zero_from_h0_component(self):
        """Test that decode_order_zero uses h_0 component correctly."""
        hidden_dim = 500
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
            },
            seed=42
        )

        # Create simple graph
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 1], dtype=torch.long)
        )

        result = encoder.forward(data)

        # Decode from full composite embedding
        constraints_full = encoder.decode_order_zero(result['graph_embedding'])

        # Decode from h_0 component directly (should give same result)
        h_0_extended = torch.zeros(3 * hidden_dim, dtype=torch.float64)
        h_0_extended[:hidden_dim] = result['h_0'].squeeze()
        constraints_h0 = encoder.decode_order_zero(h_0_extended)

        # Should have same structure
        assert len(constraints_full) == len(constraints_h0)

    def test_decode_order_one_basic(self):
        """Test basic edge decoding from composite embedding."""
        hidden_dim = 1000
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
            },
            seed=42
        )

        # Create graph with known structure
        # 2 nodes (label 0, 1), 1 edge between them
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = encoder.forward(data)
        embedding = result['graph_embedding']

        # Decode edges
        edge_constraints = encoder.decode_order_one(embedding)

        # Should detect some edges
        assert len(edge_constraints) > 0

        # Count total edges
        total_edges = sum(c['num'] for c in edge_constraints)
        assert total_edges > 0

    def test_decode_order_one_without_node_constraints(self):
        """Test that decode_order_one can work without pre-computed node constraints."""
        hidden_dim = 500
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
            },
            seed=42
        )

        # Create simple graph
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = encoder.forward(data)
        embedding = result['graph_embedding']

        # Decode edges without providing node constraints
        # (method should compute them internally)
        edge_constraints = encoder.decode_order_one(embedding, constraints_order_zero=None)

        # Should work without error
        assert isinstance(edge_constraints, list)


class TestCompositeHyperNetEdgeCases:
    """Test edge cases and special situations."""

    def test_single_node_graph(self):
        """Test encoding of a graph with single node and no edges."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Single node, no edges
        data = Data(
            x=torch.zeros(1, hidden_dim),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            batch=torch.tensor([0], dtype=torch.long),
            node_label=torch.tensor([0], dtype=torch.long)
        )

        result = encoder.forward(data)

        # Check shapes are correct
        assert result['graph_embedding'].shape == (1, 3 * hidden_dim)
        assert result['h_0'].shape == (1, hidden_dim)
        assert result['h_1'].shape == (1, hidden_dim)
        assert result['g'].shape == (1, hidden_dim)

        # h_1 should be close to zero (no edges)
        assert torch.allclose(result['h_1'], torch.zeros_like(result['h_1']), atol=1e-5)

    def test_graph_without_edges(self):
        """Test encoding of a graph with multiple nodes but no edges."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # 3 nodes, no edges
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        result = encoder.forward(data)

        # h_0 should contain node information
        assert not torch.allclose(result['h_0'], torch.zeros_like(result['h_0']))

        # h_1 should be close to zero (no edges)
        assert torch.allclose(result['h_1'], torch.zeros_like(result['h_1']), atol=1e-5)

        # g should still have some information (from nodes, even without message passing)
        assert not torch.allclose(result['g'], torch.zeros_like(result['g']))

    def test_bidirectional_edges(self):
        """Test that bidirectional flag works correctly."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            bidirectional=True,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Create graph with single directed edge
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = encoder.forward(data)

        # Should work without error
        assert result['graph_embedding'].shape == (1, 3 * hidden_dim)

    def test_decode_with_2d_embedding_stack(self):
        """Test decoding from embedding stack (2D tensor)."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
            },
            seed=42
        )

        # Create simple graph
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = encoder.forward(data)

        # Test decoding from graph_hv_stack (2D)
        embedding_stack = result['graph_hv_stack']  # Shape: (1, depth+1, 3*hidden_dim)

        # Squeeze batch dimension to get (depth+1, 3*hidden_dim)
        embedding_2d = embedding_stack.squeeze(0)

        # Should work with 2D input (uses first layer)
        node_constraints = encoder.decode_order_zero(embedding_2d)
        edge_constraints = encoder.decode_order_one(embedding_2d)

        assert isinstance(node_constraints, list)
        assert isinstance(edge_constraints, list)


class TestCompositeHyperNetDistanceExtraction:
    """Test distance embedding extraction for reconstruction."""

    def test_extract_distance_embedding_1d(self):
        """Test extracting g component from 1D composite embedding."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Create test graph
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = encoder.forward(data)
        composite_embedding = result['graph_embedding'].squeeze()  # 1D: (3*hidden_dim,)

        # Extract distance component (should be g)
        distance_embedding = encoder.extract_distance_embedding(composite_embedding)

        # Should be same as g component
        g_component = result['g'].squeeze()
        assert torch.allclose(distance_embedding, g_component, atol=1e-6)

        # Should have correct shape
        assert distance_embedding.shape == (hidden_dim,)

    def test_extract_distance_embedding_2d(self):
        """Test extracting g component from 2D batch of embeddings."""
        hidden_dim = 100
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Create batch with 2 graphs
        data = Data(
            x=torch.zeros(5, hidden_dim),
            edge_index=torch.tensor([[0, 2, 3], [1, 3, 4]], dtype=torch.long),
            batch=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            node_label=torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)
        )

        result = encoder.forward(data)
        composite_embedding = result['graph_embedding']  # 2D: (2, 3*hidden_dim)

        # Extract distance component
        distance_embedding = encoder.extract_distance_embedding(composite_embedding)

        # Should match g component
        g_component = result['g']
        assert torch.allclose(distance_embedding, g_component, atol=1e-6)

        # Should have correct shape
        assert distance_embedding.shape == (2, hidden_dim)

    def test_hypernet_distance_extraction(self):
        """Test that standard HyperNet returns full embedding."""
        hidden_dim = 100
        encoder = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Create test graph
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = encoder.forward(data)
        embedding = result['graph_embedding']

        # Extract distance component (should be full embedding for HyperNet)
        distance_embedding = encoder.extract_distance_embedding(embedding)

        # Should be identical
        assert torch.allclose(distance_embedding, embedding)

    def test_distance_embedding_for_reconstruction(self):
        """Test that distance embedding works for reconstruction scenario."""
        hidden_dim = 500
        encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
            },
            seed=42
        )

        # Encode original graph
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        result = encoder.forward(data)
        target_embedding = result['graph_embedding']

        # Simulate reconstruction: encode candidate graph
        candidate_data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        candidate_result = encoder.forward(candidate_data)
        candidate_embedding = candidate_result['graph_embedding']

        # Extract distance components (should use only g)
        target_g = encoder.extract_distance_embedding(target_embedding)
        candidate_g = encoder.extract_distance_embedding(candidate_embedding)

        # Compute distance on g components
        from graph_hdc.reconstruct import cosine_distance
        distance = cosine_distance(target_g, candidate_g)

        # Should be very small (same graph)
        assert distance < 0.01

        print(f"Distance between identical graphs (g component only): {distance:.6f}")


class TestCompositeHyperNetComparison:
    """Compare CompositeHyperNet with standard HyperNet."""

    def test_comparison_with_hypernet(self):
        """Compare embedding sizes between HyperNet and CompositeHyperNet."""
        hidden_dim = 100
        node_encoder_map = {
            'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
        }

        # Create both encoders
        hypernet = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
            seed=42
        )

        composite_hypernet = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map=node_encoder_map,
            seed=42
        )

        # Create test graph
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        # Forward pass through both
        result_hyper = hypernet.forward(data)
        result_composite = composite_hypernet.forward(data)

        # CompositeHyperNet embedding should be 3x larger
        assert result_hyper['graph_embedding'].shape == (1, hidden_dim)
        assert result_composite['graph_embedding'].shape == (1, 3 * hidden_dim)

    def test_decode_accuracy_comparison(self):
        """Test that CompositeHyperNet decoding might be more accurate."""
        hidden_dim = 2000  # Large dimension for better accuracy
        node_encoder_map = {
            'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5, seed=42)
        }

        # Create CompositeHyperNet
        composite_encoder = CompositeHyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map=node_encoder_map,
            seed=42
        )

        # Create test graph with known structure
        # 2 nodes of label 0, 1 node of label 1, 1 node of label 2
        data = Data(
            x=torch.zeros(4, hidden_dim),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 0, 1, 2], dtype=torch.long)
        )

        result = composite_encoder.forward(data)
        embedding = result['graph_embedding']

        # Decode nodes
        node_constraints = composite_encoder.decode_order_zero(embedding)

        # Count decoded nodes per label
        label_counts = {}
        for constraint in node_constraints:
            label = constraint['src']['node_label']
            label_counts[label] = label_counts.get(label, 0) + constraint['num']

        # Total should be 4
        total = sum(label_counts.values())
        assert total == 4, f"Expected 4 total nodes, got {total}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
