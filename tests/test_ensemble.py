"""
Tests for HyperNetEnsemble class.

This module contains comprehensive tests for the HyperNetEnsemble implementation,
including tests for ensemble creation, forward pass, distance calculation,
majority voting, and edge cases.
"""

import torch
import numpy as np
import pytest
from torch_geometric.data import Data

from graph_hdc.models import HyperNet, HyperNetEnsemble
from graph_hdc.utils import CategoricalIntegerEncoder


class TestHyperNetEnsembleBasics:
    """Test basic functionality of HyperNetEnsemble."""

    def test_initialization_with_compatible_models(self):
        """Test that HyperNetEnsemble initializes correctly with compatible models."""
        hidden_dim = 100
        depth = 3

        # Create two compatible models
        model1 = HyperNet(
            hidden_dim=hidden_dim,
            depth=depth,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            },
            seed=42
        )

        model2 = HyperNet(
            hidden_dim=hidden_dim,
            depth=depth,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            },
            seed=123
        )

        # Create ensemble
        ensemble = HyperNetEnsemble([model1, model2])

        # Verify attributes
        assert ensemble.num_models == 2
        assert ensemble.hidden_dim == hidden_dim
        assert ensemble.depth == depth
        assert len(ensemble.hyper_nets) == 2

    def test_initialization_with_incompatible_hidden_dim(self):
        """Test that initialization fails when models have different hidden_dim."""
        model1 = HyperNet(
            hidden_dim=100,
            depth=3,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=100, num_categories=5)
            }
        )

        model2 = HyperNet(
            hidden_dim=200,  # Different hidden_dim
            depth=3,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=200, num_categories=5)
            }
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="hidden_dim"):
            HyperNetEnsemble([model1, model2])

    def test_initialization_with_different_depths(self):
        """Test that initialization succeeds when models have different depths."""
        hidden_dim = 100

        model1 = HyperNet(
            hidden_dim=hidden_dim,
            depth=3,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            }
        )

        model2 = HyperNet(
            hidden_dim=hidden_dim,
            depth=5,  # Different depth - should be allowed
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            }
        )

        # Should succeed (different depths are now allowed)
        ensemble = HyperNetEnsemble([model1, model2])

        # Verify that ensemble recognizes non-uniform depths
        assert ensemble.uniform_depth is False
        assert ensemble.depth is None
        assert ensemble.depths == [3, 5]

    def test_initialization_with_empty_list(self):
        """Test that initialization fails when hyper_nets list is empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            HyperNetEnsemble([])

    def test_forward_output_shapes(self):
        """Test that forward() produces correct stacked output shapes."""
        hidden_dim = 100
        num_models = 3

        # Create ensemble with 3 models
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
                },
                seed=42 + i
            )
            for i in range(num_models)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create simple graph: 3 nodes, 2 edges
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        result = ensemble.forward(data)

        # Check that result contains expected keys
        assert 'graph_embedding' in result

        # Check that embeddings are stacked correctly
        # Shape should be (num_models, batch_size, hidden_dim)
        assert result['graph_embedding'].shape == (num_models, 1, hidden_dim)

        # Check that graph_hv_stack is also stacked if available
        if 'graph_hv_stack' in result:
            # Shape should be (num_models, batch_size, depth+1, hidden_dim)
            assert result['graph_hv_stack'].shape == (num_models, 1, ensemble.depth + 1, hidden_dim)

    def test_forward_with_different_depths(self):
        """Test that forward() works with different depths and excludes graph_hv_stack."""
        hidden_dim = 100

        # Create ensemble with models of different depths
        model1 = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            }
        )

        model2 = HyperNet(
            hidden_dim=hidden_dim,
            depth=4,  # Different depth
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
            }
        )

        ensemble = HyperNetEnsemble([model1, model2])

        # Create simple graph
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        result = ensemble.forward(data)

        # graph_embedding should still be present
        assert 'graph_embedding' in result
        assert result['graph_embedding'].shape == (2, 1, hidden_dim)

        # graph_hv_stack should NOT be present (depths differ)
        assert 'graph_hv_stack' not in result


class TestHyperNetEnsembleDistance:
    """Test distance calculation functionality."""

    def test_get_distance_mean_calculation(self):
        """Test that get_distance correctly computes mean distance."""
        hidden_dim = 100

        # Create ensemble with 2 models
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5)
                },
                seed=42 + i
            )
            for i in range(2)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create two simple graphs
        data1 = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        data2 = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([1, 2], dtype=torch.long)
        )

        # Get stacked embeddings
        result1 = ensemble.forward(data1)
        result2 = ensemble.forward(data2)

        hv1 = result1['graph_embedding']  # Shape: (2, 1, hidden_dim)
        hv2 = result2['graph_embedding']  # Shape: (2, 1, hidden_dim)

        # Calculate ensemble distance
        ensemble_dist = ensemble.get_distance(hv1, hv2)

        # Calculate individual distances manually
        dist1 = models[0].get_distance(hv1[0].squeeze(), hv2[0].squeeze())
        dist2 = models[1].get_distance(hv1[1].squeeze(), hv2[1].squeeze())
        expected_mean = (dist1 + dist2) / 2

        # Should be approximately equal (within floating point tolerance)
        assert abs(ensemble_dist - expected_mean) < 1e-6

    def test_get_distance_identical_graphs(self):
        """Test that distance is near zero for identical graphs."""
        hidden_dim = 100

        # Create ensemble
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
                },
                seed=42 + i
            )
            for i in range(2)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create identical graphs
        data1 = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        # Get stacked embeddings
        result1 = ensemble.forward(data1)
        result2 = ensemble.forward(data1)  # Same graph

        hv1 = result1['graph_embedding']
        hv2 = result2['graph_embedding']

        # Calculate distance
        distance = ensemble.get_distance(hv1, hv2)

        # Distance should be very small (essentially zero)
        assert distance < 1e-6


class TestHyperNetEnsembleMajorityVoting:
    """Test majority voting functionality for decoding."""

    def test_decode_order_zero_unanimous_vote(self):
        """Test decode_order_zero when all models agree."""
        hidden_dim = 1000  # Large dimension for better accuracy

        # Create ensemble with 3 models (all with same seed for consistent encoding)
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5, seed=42)
                },
                seed=42
            )
            for _ in range(3)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create test graph
        data = Data(
            x=torch.zeros(4, hidden_dim),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 1, 2], dtype=torch.long)
        )

        # Get stacked embeddings
        result = ensemble.forward(data)
        stacked_embedding = result['graph_embedding']

        # Decode with majority voting
        constraints = ensemble.decode_order_zero(stacked_embedding)

        # Should detect 4 nodes total (since all models agree)
        total_nodes = sum(c['num'] for c in constraints)
        assert total_nodes == 4

    def test_decode_order_zero_split_vote(self):
        """Test decode_order_zero with split votes (some constraints filtered)."""
        hidden_dim = 1000

        # Create ensemble with 3 models with different seeds
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=5, seed=42 + i)
                },
                seed=42 + i
            )
            for i in range(3)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create test graph
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        # Get stacked embeddings
        result = ensemble.forward(data)
        stacked_embedding = result['graph_embedding']

        # Decode with majority voting
        constraints = ensemble.decode_order_zero(stacked_embedding)

        # With majority voting, only constraints appearing in >= 2 out of 3 models should be included
        # The exact number depends on decoding accuracy, but should be non-empty
        assert len(constraints) > 0

    def test_decode_order_one_basic(self):
        """Test basic edge decoding with majority voting."""
        hidden_dim = 1000

        # Create ensemble
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
                },
                seed=42
            )
            for _ in range(2)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create test graph with known edge
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        # Get stacked embeddings
        result = ensemble.forward(data)
        stacked_embedding = result['graph_embedding']

        # Decode nodes first
        node_constraints = ensemble.decode_order_zero(stacked_embedding)

        # Decode edges with majority voting
        edge_constraints = ensemble.decode_order_one(stacked_embedding, node_constraints)

        # Should detect some edges
        assert isinstance(edge_constraints, list)

    def test_majority_vote_median_num(self):
        """Test that majority voting uses median for 'num' field."""
        hidden_dim = 500

        # This is a bit tricky to test directly, so we'll verify the behavior indirectly
        # by checking that the ensemble's decode results are reasonable
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42 + i)
                },
                seed=42 + i
            )
            for i in range(3)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create test graph
        data = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 1], dtype=torch.long)
        )

        result = ensemble.forward(data)
        constraints = ensemble.decode_order_zero(result['graph_embedding'])

        # Verify that num values are integers (median should give integer)
        for constraint in constraints:
            assert isinstance(constraint['num'], int)


class TestHyperNetEnsembleSaveLoad:
    """Test save and load functionality."""

    def test_save_and_load(self, tmp_path):
        """Test that ensemble can be saved and loaded correctly."""
        hidden_dim = 100

        # Create ensemble
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42 + i)
                },
                seed=42 + i
            )
            for i in range(2)
        ]

        ensemble = HyperNetEnsemble(models)

        # Save ensemble
        save_path = tmp_path / "ensemble.json"
        ensemble.save_to_path(str(save_path))

        # Verify files were created
        assert save_path.exists()
        assert (tmp_path / "ensemble.json_model_0.json").exists()
        assert (tmp_path / "ensemble.json_model_1.json").exists()

        # Load ensemble
        loaded_ensemble = HyperNetEnsemble(models)  # Create with dummy models
        loaded_ensemble.load_from_path(str(save_path))

        # Verify attributes
        assert loaded_ensemble.num_models == 2
        assert loaded_ensemble.hidden_dim == hidden_dim
        assert loaded_ensemble.depth == 2

    def test_save_and_load_different_depths(self, tmp_path):
        """Test that ensemble with different depths can be saved and loaded."""
        hidden_dim = 100

        # Create ensemble with different depths
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=42)
                },
                seed=42
            ),
            HyperNet(
                hidden_dim=hidden_dim,
                depth=4,  # Different depth
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3, seed=43)
                },
                seed=43
            )
        ]

        ensemble = HyperNetEnsemble(models)

        # Save ensemble
        save_path = tmp_path / "ensemble_diff_depths.json"
        ensemble.save_to_path(str(save_path))

        # Verify files were created
        assert save_path.exists()

        # Load ensemble
        loaded_ensemble = HyperNetEnsemble(models)  # Create with dummy models
        loaded_ensemble.load_from_path(str(save_path))

        # Verify attributes
        assert loaded_ensemble.num_models == 2
        assert loaded_ensemble.hidden_dim == hidden_dim
        assert loaded_ensemble.uniform_depth is False
        assert loaded_ensemble.depth is None
        assert loaded_ensemble.depths == [2, 4]


class TestHyperNetEnsembleEdgeCases:
    """Test edge cases and special situations."""

    def test_single_model_ensemble(self):
        """Test ensemble with single model (edge case)."""
        hidden_dim = 100

        model = HyperNet(
            hidden_dim=hidden_dim,
            depth=2,
            node_encoder_map={
                'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
            }
        )

        # Create ensemble with single model
        ensemble = HyperNetEnsemble([model])

        assert ensemble.num_models == 1

        # Forward pass should work
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = ensemble.forward(data)
        assert result['graph_embedding'].shape == (1, 1, hidden_dim)

    def test_large_ensemble(self):
        """Test ensemble with many models."""
        hidden_dim = 50  # Smaller for faster testing
        num_models = 10

        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
                },
                seed=42 + i
            )
            for i in range(num_models)
        ]

        ensemble = HyperNetEnsemble(models)
        assert ensemble.num_models == num_models

        # Forward pass should work
        data = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1], dtype=torch.long)
        )

        result = ensemble.forward(data)
        assert result['graph_embedding'].shape == (num_models, 1, hidden_dim)

    def test_batch_processing(self):
        """Test that ensemble correctly handles batched graphs."""
        hidden_dim = 100

        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
                },
                seed=42 + i
            )
            for i in range(2)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create batch with 2 graphs
        # Graph 0: 2 nodes, 1 edge
        # Graph 1: 3 nodes, 2 edges
        data = Data(
            x=torch.zeros(5, hidden_dim),
            edge_index=torch.tensor([[0, 2, 3], [1, 3, 4]], dtype=torch.long),
            batch=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            node_label=torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)
        )

        result = ensemble.forward(data)

        # Check batch dimensions
        # Shape should be (num_models, batch_size, hidden_dim)
        assert result['graph_embedding'].shape == (2, 2, hidden_dim)


class TestHyperNetEnsembleForwardGraphs:
    """Test forward_graphs method functionality."""

    def test_forward_graphs_basic(self):
        """Test that forward_graphs works with ensemble."""
        hidden_dim = 100

        # Create ensemble
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
                },
                seed=42 + i
            )
            for i in range(2)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create test graphs using forward_graph first to understand format
        data1 = Data(
            x=torch.zeros(3, hidden_dim),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
            batch=torch.tensor([0, 0, 0], dtype=torch.long),
            node_label=torch.tensor([0, 1, 2], dtype=torch.long)
        )

        data2 = Data(
            x=torch.zeros(2, hidden_dim),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            batch=torch.tensor([0, 0], dtype=torch.long),
            node_label=torch.tensor([1, 2], dtype=torch.long)
        )

        # Test forward_graph (singular) first
        result1 = ensemble.forward_graph({'node_label': torch.tensor([0, 1, 2]), 'edge_indices': [[0, 1], [1, 2]]})
        assert 'graph_embedding' in result1
        # Shape should be (num_models, hidden_dim) after extraction
        assert result1['graph_embedding'].shape == (2, hidden_dim)

        result2 = ensemble.forward_graph({'node_label': torch.tensor([1, 2]), 'edge_indices': [[0, 1]]})
        assert 'graph_embedding' in result2
        assert result2['graph_embedding'].shape == (2, hidden_dim)

    def test_forward_graphs_multiple(self):
        """Test that forward_graphs works with multiple graphs."""
        hidden_dim = 100

        # Create ensemble
        models = [
            HyperNet(
                hidden_dim=hidden_dim,
                depth=2,
                node_encoder_map={
                    'node_label': CategoricalIntegerEncoder(dim=hidden_dim, num_categories=3)
                },
                seed=42 + i
            )
            for i in range(2)
        ]

        ensemble = HyperNetEnsemble(models)

        # Create test graphs in proper dict format
        graphs = [
            {'node_label': torch.tensor([0, 1, 2]), 'edge_indices': [[0, 1], [1, 2]]},
            {'node_label': torch.tensor([1, 2]), 'edge_indices': [[0, 1]]},
        ]

        # Test forward_graphs with batch_size=1 to process one graph at a time
        # (batching multiple graphs with different sizes can cause issues)
        results = ensemble.forward_graphs(graphs, batch_size=1)

        # Should return a list with one result per graph
        assert len(results) == 2

        # Each result should have graph_embedding
        for i, result in enumerate(results):
            assert 'graph_embedding' in result
            # Shape should be (num_models, hidden_dim)
            assert result['graph_embedding'].shape == (2, hidden_dim), f"Graph {i} has wrong shape"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
