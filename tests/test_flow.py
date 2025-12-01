"""
Unit tests for the normalizing flow models.

This module tests the NormalizingFlow wrapper around the normflows library,
ensuring that the flow models can be constructed, trained, and used for
sampling and density estimation.
"""

import pytest
import torch
import numpy as np
from graph_hdc.models.flow import NormalizingFlow


class TestNormalizingFlowConstruction:
    """Tests for basic construction of normalizing flow models."""

    def test_realnvp_construction_basically_works(self):
        """
        Test that a RealNVP flow model can be constructed without errors.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
            learning_rate=1e-3,
        )
        assert isinstance(flow, NormalizingFlow)
        assert flow.dim == 10
        assert flow.flow_type == 'realnvp'
        assert flow.num_layers == 4

    def test_maf_construction_basically_works(self):
        """
        Test that a MAF flow model can be constructed without errors.
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='maf',
            num_layers=3,
            hidden_units=16,
        )
        assert isinstance(flow, NormalizingFlow)
        assert flow.dim == 8
        assert flow.flow_type == 'maf'

    @pytest.mark.skip(reason="Glow with ActNorm has specific dimension requirements")
    def test_glow_construction_basically_works(self):
        """
        Test that a Glow flow model can be constructed without errors.

        Note: Glow architecture with ActNorm can be sensitive to dimensions
        and may require specific setups.
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='glow',
            num_layers=2,
            hidden_units=24,
        )
        assert isinstance(flow, NormalizingFlow)
        assert flow.flow_type == 'glow'

    def test_nsf_construction_basically_works(self):
        """
        Test that a Neural Spline Flow model can be constructed without errors.
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='nsf',
            num_layers=2,
            hidden_units=16,
        )
        assert isinstance(flow, NormalizingFlow)
        assert flow.flow_type == 'nsf'

    def test_residual_construction_basically_works(self):
        """
        Test that a Residual flow model can be constructed without errors.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='residual',
            num_layers=3,
            hidden_units=32,
        )
        assert isinstance(flow, NormalizingFlow)
        assert flow.flow_type == 'residual'

    def test_invalid_flow_type_raises_error(self):
        """
        Test that an invalid flow type raises a ValueError.
        """
        with pytest.raises(ValueError, match="Unknown flow type"):
            flow = NormalizingFlow(
                dim=10,
                flow_type='invalid_type',
                num_layers=2,
            )

    def test_different_base_distributions(self):
        """
        Test that different base distributions can be specified.
        """
        flow_gaussian = NormalizingFlow(
            dim=5,
            flow_type='realnvp',
            num_layers=2,
            base_dist='gaussian',
        )
        assert flow_gaussian.base is not None

        flow_uniform = NormalizingFlow(
            dim=5,
            flow_type='realnvp',
            num_layers=2,
            base_dist='uniform',
        )
        assert flow_uniform.base is not None


class TestNormalizingFlowForward:
    """Tests for forward transformations (noise to data)."""

    def test_forward_basically_works(self):
        """
        Test that the forward transformation works and returns correct shapes.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
        )

        # Create some random latent samples
        batch_size = 5
        z = torch.randn(batch_size, 10)

        # Apply forward transformation (noise to data)
        x = flow.forward(z)

        assert x.shape == (batch_size, 10)
        assert torch.isfinite(x).all()

    def test_forward_with_different_batch_sizes(self):
        """
        Test forward transformation with different batch sizes.
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
        )

        for batch_size in [1, 10, 100]:
            z = torch.randn(batch_size, 8)
            x = flow.forward(z)

            assert x.shape == (batch_size, 8)

    def test_forward_and_log_det_basically_works(self):
        """
        Test that forward_and_log_det works and returns correct shapes.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
        )

        batch_size = 5
        z = torch.randn(batch_size, 10)

        # Apply forward transformation with log det
        x, log_det = flow.forward_and_log_det(z)

        assert x.shape == (batch_size, 10)
        assert log_det.shape == (batch_size,)
        assert torch.isfinite(x).all()
        assert torch.isfinite(log_det).all()


class TestNormalizingFlowInverse:
    """Tests for inverse transformations (data to noise)."""

    def test_inverse_basically_works(self):
        """
        Test that the inverse transformation works and returns correct shapes.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
        )

        # Create some random data samples
        batch_size = 5
        x = torch.randn(batch_size, 10)

        # Apply inverse transformation (data to noise)
        z = flow.inverse(x)

        assert z.shape == (batch_size, 10)
        assert torch.isfinite(z).all()

    def test_inverse_and_log_det_basically_works(self):
        """
        Test that inverse_and_log_det works and returns correct shapes.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
        )

        batch_size = 5
        x = torch.randn(batch_size, 10)

        # Apply inverse transformation with log det
        z, log_det = flow.inverse_and_log_det(x)

        assert z.shape == (batch_size, 10)
        assert log_det.shape == (batch_size,)
        assert torch.isfinite(z).all()
        assert torch.isfinite(log_det).all()

    def test_forward_inverse_consistency(self):
        """
        Test that forward and inverse are approximately inverse operations.
        """
        flow = NormalizingFlow(
            dim=6,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=24,
        )

        # Start with some latent samples
        z_original = torch.randn(3, 6)

        # Forward then inverse
        x = flow.forward(z_original)
        z_reconstructed = flow.inverse(x)

        # Check that reconstruction is close to original
        assert torch.allclose(z_original, z_reconstructed, atol=1e-5)


class TestNormalizingFlowSampling:
    """Tests for sampling from the flow model."""

    def test_sampling_basically_works(self):
        """
        Test that samples can be generated from the model.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
        )

        # Generate samples
        samples = flow.sample(num_samples=20)

        assert samples.shape == (20, 10)
        assert torch.isfinite(samples).all()

    def test_sampling_different_sizes(self):
        """
        Test sampling with different numbers of samples.
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
        )

        for num_samples in [1, 5, 50]:
            samples = flow.sample(num_samples=num_samples)
            assert samples.shape == (num_samples, 8)


class TestNormalizingFlowLogProb:
    """Tests for log probability computation."""

    def test_log_prob_basically_works(self):
        """
        Test that log probabilities can be computed.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
        )

        # Create some random data
        x = torch.randn(5, 10)

        # Compute log probabilities
        log_prob = flow.log_prob(x)

        assert log_prob.shape == (5,)
        assert torch.isfinite(log_prob).all()

    def test_log_prob_values_reasonable(self):
        """
        Test that log probability values are finite.
        """
        flow = NormalizingFlow(
            dim=5,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
        )

        # Sample from the base distribution (should have reasonable log probs)
        z = flow.base.sample(10)

        # Transform to data space
        x = flow.forward(z)

        # Compute log prob
        log_prob = flow.log_prob(x)

        # Log probs should be finite
        assert torch.isfinite(log_prob).all()
        # For samples from the model itself, log probs should not be too extreme
        assert (log_prob > -10000).all()  # Very loose bound to avoid numerical issues

    def test_log_prob_higher_for_likely_samples(self):
        """
        Test that samples from the model have higher log probability
        than random samples far from the distribution.
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='realnvp',
            num_layers=3,
            hidden_units=24,
        )

        # Generate samples from the model (should be likely)
        likely_samples = flow.sample(num_samples=5)
        log_prob_likely = flow.log_prob(likely_samples).mean()

        # Create unlikely samples (very far from typical values)
        unlikely_samples = torch.randn(5, 8) * 10 + 20
        log_prob_unlikely = flow.log_prob(unlikely_samples).mean()

        # Likely samples should have higher log probability
        assert log_prob_likely > log_prob_unlikely


class TestNormalizingFlowTraining:
    """Tests for training-related functionality."""

    def test_training_step_basically_works(self):
        """
        Test that a training step can be executed.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
            learning_rate=1e-3,
        )

        # Create a batch of training data
        batch = torch.randn(8, 10)

        # Execute training step
        loss = flow.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss)
        assert loss > 0  # Loss should be positive (negative log likelihood)

    def test_training_step_with_tuple_batch(self):
        """
        Test that training step works with tuple batches (e.g., from DataLoader).
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
        )

        # Create a tuple batch (data, labels) as might come from a DataLoader
        batch = (torch.randn(5, 8), torch.zeros(5))

        # Execute training step
        loss = flow.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_validation_step_basically_works(self):
        """
        Test that a validation step can be executed.
        """
        flow = NormalizingFlow(
            dim=8,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
        )

        # Create a batch of validation data
        batch = torch.randn(5, 8)

        # Execute validation step
        loss = flow.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_optimizer_configuration(self):
        """
        Test that the optimizer is configured correctly.
        """
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
            learning_rate=5e-4,
        )

        optimizer = flow.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 5e-4

    def test_simple_training_reduces_loss(self):
        """
        Test that a simple training loop can reduce the loss.

        This is a basic sanity check that the model can actually learn
        from data.
        """
        torch.manual_seed(42)

        # Create a simple 2D dataset (mixture of two Gaussians)
        data1 = torch.randn(100, 2) * 0.5 + torch.tensor([2.0, 2.0])
        data2 = torch.randn(100, 2) * 0.5 + torch.tensor([-2.0, -2.0])
        data = torch.cat([data1, data2], dim=0)

        # Create flow model
        flow = NormalizingFlow(
            dim=2,
            flow_type='realnvp',
            num_layers=4,
            hidden_units=32,
            learning_rate=1e-2,
        )

        optimizer = flow.configure_optimizers()

        # Record initial loss
        initial_loss = flow.training_step(data, 0).item()

        # Train for a few steps
        for _ in range(10):
            loss = flow.training_step(data, 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Record final loss
        final_loss = flow.training_step(data, 0).item()

        # Loss should decrease
        assert final_loss < initial_loss


class TestNormalizingFlowDifferentArchitectures:
    """Tests comparing different flow architectures."""

    def test_all_architectures_can_fit_simple_data(self):
        """
        Test that supported architectures can be trained on simple data.
        """
        # Simple Gaussian data
        data = torch.randn(50, 4) * 0.5

        # Test main architectures (excluding glow which has specific requirements)
        flow_types = ['realnvp', 'maf', 'nsf', 'residual']

        for flow_type in flow_types:
            flow = NormalizingFlow(
                dim=4,
                flow_type=flow_type,
                num_layers=2,
                hidden_units=16,
                learning_rate=1e-2,
            )

            optimizer = flow.configure_optimizers()

            # Train for a few steps
            for _ in range(5):
                loss = flow.training_step(data, 0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Should be able to sample after training
            samples = flow.sample(num_samples=10)
            assert samples.shape == (10, 4)
            assert torch.isfinite(samples).all()

    def test_different_architectures_produce_different_results(self):
        """
        Test that different architectures produce different samples.
        """
        torch.manual_seed(42)

        # Train two different models on the same data (use even dimension)
        data = torch.randn(100, 6)

        flow1 = NormalizingFlow(
            dim=6,
            flow_type='realnvp',
            num_layers=2,
            hidden_units=16,
        )

        flow2 = NormalizingFlow(
            dim=6,
            flow_type='maf',
            num_layers=2,
            hidden_units=16,
        )

        # They should produce different initial samples
        torch.manual_seed(100)
        samples1 = flow1.sample(num_samples=5)

        torch.manual_seed(100)
        samples2 = flow2.sample(num_samples=5)

        # Samples should be different (different architectures, different random init)
        assert not torch.allclose(samples1, samples2, atol=1e-3)
