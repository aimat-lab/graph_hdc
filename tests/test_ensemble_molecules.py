"""
Test HyperNetEnsemble with molecular graphs from SMILES strings.

This module tests that the ensemble works correctly with real molecular graphs.
"""

import pytest
from rdkit import Chem

from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map
from graph_hdc.models import HyperNet, HyperNetEnsemble


class TestHyperNetEnsembleWithMolecules:
    """Test ensemble with molecular graphs."""

    def test_forward_graph_single_molecule(self):
        """Test forward_graph with a single molecule."""
        # Create molecule from SMILES
        smiles = 'CCO'  # Ethanol
        mol = Chem.MolFromSmiles(smiles)
        graph = graph_dict_from_mol(mol)

        # Create ensemble
        encoder_map = make_molecule_node_encoder_map(dim=100, seed=42)
        model1 = HyperNet(hidden_dim=100, depth=2, node_encoder_map=encoder_map, seed=42)
        model2 = HyperNet(hidden_dim=100, depth=3, node_encoder_map=encoder_map, seed=123)
        ensemble = HyperNetEnsemble([model1, model2])

        # Test forward_graph
        result = ensemble.forward_graph(graph)

        # Verify result structure
        assert 'graph_embedding' in result
        # Shape should be (num_models, hidden_dim) = (2, 100)
        assert result['graph_embedding'].shape == (2, 100)

    def test_forward_graphs_multiple_molecules(self):
        """Test forward_graphs with multiple molecules."""
        # Create multiple molecules
        smiles_list = ['CCO', 'CC', 'CCC']  # Ethanol, Ethane, Propane
        graphs = [graph_dict_from_mol(Chem.MolFromSmiles(s)) for s in smiles_list]

        # Create ensemble
        encoder_map = make_molecule_node_encoder_map(dim=100, seed=42)
        model1 = HyperNet(hidden_dim=100, depth=2, node_encoder_map=encoder_map, seed=42)
        model2 = HyperNet(hidden_dim=100, depth=3, node_encoder_map=encoder_map, seed=123)
        ensemble = HyperNetEnsemble([model1, model2])

        # Test forward_graphs
        result_list = ensemble.forward_graphs(graphs)

        # Should return one result per graph
        assert len(result_list) == len(graphs)

        # Each result should have stacked embeddings
        for i, result in enumerate(result_list):
            assert 'graph_embedding' in result
            # Each graph gets stacked embeddings: (num_models, hidden_dim)
            assert result['graph_embedding'].shape == (2, 100)

    def test_decode_from_forward_graph_result(self):
        """Test that decoding works with forward_graph results."""
        # Create molecule
        smiles = 'CCO'
        mol = Chem.MolFromSmiles(smiles)
        graph = graph_dict_from_mol(mol)

        # Create ensemble
        encoder_map = make_molecule_node_encoder_map(dim=1000, seed=42)
        model1 = HyperNet(hidden_dim=1000, depth=2, node_encoder_map=encoder_map, seed=42)
        model2 = HyperNet(hidden_dim=1000, depth=3, node_encoder_map=encoder_map, seed=123)
        ensemble = HyperNetEnsemble([model1, model2])

        # Get embedding
        result = ensemble.forward_graph(graph)
        embedding = result['graph_embedding']

        # Decode nodes - should work with stacked embeddings
        node_constraints = ensemble.decode_order_zero(embedding)
        assert isinstance(node_constraints, list)
        assert len(node_constraints) > 0

        # Total nodes should be approximately 3 (C-C-O)
        # HDC is probabilistic, so we allow some tolerance
        total_nodes = sum(c['num'] for c in node_constraints)
        assert 2 <= total_nodes <= 4  # Allow small variations

    def test_decode_from_forward_graphs_results(self):
        """Test that decoding works with forward_graphs results."""
        # Create multiple molecules
        smiles_list = ['CCO', 'CC']
        graphs = [graph_dict_from_mol(Chem.MolFromSmiles(s)) for s in smiles_list]

        # Create ensemble
        encoder_map = make_molecule_node_encoder_map(dim=1000, seed=42)
        model1 = HyperNet(hidden_dim=1000, depth=2, node_encoder_map=encoder_map, seed=42)
        model2 = HyperNet(hidden_dim=1000, depth=3, node_encoder_map=encoder_map, seed=123)
        ensemble = HyperNetEnsemble([model1, model2])

        # Get embeddings for all graphs
        result_list = ensemble.forward_graphs(graphs)

        # Decode each graph individually
        for i, result in enumerate(result_list):
            embedding = result['graph_embedding']

            # Decode nodes
            node_constraints = ensemble.decode_order_zero(embedding)
            assert isinstance(node_constraints, list)
            assert len(node_constraints) > 0

            # Verify approximately correct number of nodes
            # HDC is probabilistic, so we allow some tolerance
            total_nodes = sum(c['num'] for c in node_constraints)
            expected_nodes = len(graphs[i]['node_indices'])
            assert expected_nodes - 1 <= total_nodes <= expected_nodes + 1

    def test_ensemble_vs_individual_models(self):
        """Compare ensemble forward_graph with individual model results."""
        # Create molecule
        smiles = 'CCO'
        mol = Chem.MolFromSmiles(smiles)
        graph = graph_dict_from_mol(mol)

        # Create models
        encoder_map = make_molecule_node_encoder_map(dim=100, seed=42)
        model1 = HyperNet(hidden_dim=100, depth=2, node_encoder_map=encoder_map, seed=42)
        model2 = HyperNet(hidden_dim=100, depth=3, node_encoder_map=encoder_map, seed=123)

        # Get individual model results
        result1 = model1.forward_graph(graph)
        result2 = model2.forward_graph(graph)

        # Create ensemble and get result
        ensemble = HyperNetEnsemble([model1, model2])
        ensemble_result = ensemble.forward_graph(graph)

        # Ensemble result should stack both model embeddings
        assert ensemble_result['graph_embedding'].shape == (2, 100)

        # First row should match model1, second row should match model2
        import numpy as np
        assert np.allclose(ensemble_result['graph_embedding'][0], result1['graph_embedding'], atol=1e-6)
        assert np.allclose(ensemble_result['graph_embedding'][1], result2['graph_embedding'], atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
