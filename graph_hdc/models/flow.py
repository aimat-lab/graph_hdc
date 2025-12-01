"""
Normalizing flow models using the normflows library.

This module provides a PyTorch Lightning wrapper around the normflows library
for building normalizing flow models. The flows can map between target and noise
distributions bidirectionally, enabling both density estimation and sample generation.
"""

from typing import Literal, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import pytorch_lightning as pl
import normflows as nf


class NormalizingFlow(pl.LightningModule):
    """
    Normalizing flow model wrapped as a PyTorch Lightning module.

    This class provides a convenient wrapper around the normflows library's
    NormalizingFlow class, adding PyTorch Lightning training capabilities.
    It supports various flow architectures including RealNVP, MAF, Glow, and others.

    **Design Rationale**

    Normalizing flows provide exact likelihood computation and efficient sampling,
    making them suitable for generative modeling and density estimation. By wrapping
    the normflows library in a Lightning module, we get automatic training loops,
    logging, checkpointing, and other conveniences while leveraging well-tested
    flow implementations.

    **Example Usage**

    .. code-block:: python

        # Create a RealNVP flow model
        flow = NormalizingFlow(
            dim=10,
            flow_type='realnvp',
            num_layers=8,
            hidden_units=64,
            learning_rate=1e-3,
        )

        # Sample from the model
        samples = flow.sample(num_samples=100)

        # Compute log probability
        log_prob = flow.log_prob(samples)

        # Train with PyTorch Lightning
        trainer = pl.Trainer(max_epochs=100)
        trainer.fit(flow, train_dataloader)

    :param dim: Dimensionality of the input/output space
    :param flow_type: Type of flow architecture ('realnvp', 'maf', 'glow', 'nsf', 'residual')
    :param num_layers: Number of flow layers to stack
    :param hidden_units: Number of hidden units in the transformation networks
    :param hidden_layers: Number of hidden layers in each transformation network
    :param learning_rate: Learning rate for the Adam optimizer
    :param base_dist: Base distribution type ('gaussian' or 'uniform')
    """

    def __init__(
        self,
        dim: int,
        flow_type: Literal['realnvp', 'maf', 'glow', 'nsf', 'residual'] = 'realnvp',
        num_layers: int = 8,
        hidden_units: int = 64,
        hidden_layers: int = 2,
        learning_rate: float = 1e-3,
        base_dist: Literal['gaussian', 'uniform'] = 'gaussian',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dim = dim
        self.flow_type = flow_type
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        # Create base distribution
        if base_dist == 'gaussian':
            self.base = nf.distributions.DiagGaussian(dim)
        elif base_dist == 'uniform':
            # UniformGaussian with all dimensions uniform
            self.base = nf.distributions.UniformGaussian(dim, ind=list(range(dim)))
        else:
            raise ValueError(f"Unknown base distribution: {base_dist}")

        # Create flow layers
        flows = self._create_flows()

        # Create the normalizing flow model
        self.flow = nf.NormalizingFlow(self.base, flows)

    def _create_flows(self) -> List[nf.flows.Flow]:
        """
        Create the flow transformation layers based on the specified architecture.

        :returns: List of flow transformation layers
        """
        flows = []

        if self.flow_type == 'realnvp':
            # Real NVP with alternating mask
            # Create binary mask: 1 for first half, 0 for second half
            b = torch.zeros(self.dim)
            b[: self.dim // 2] = 1

            for i in range(self.num_layers):
                # Add ActNorm for numerical stability (especially important for high dimensions)
                flows.append(nf.flows.ActNorm(self.dim))

                # Scale and translation networks
                s = nf.nets.MLP(
                    [self.dim, self.hidden_units] +
                    [self.hidden_units] * (self.hidden_layers - 1) +
                    [self.dim],
                    init_zeros=True,
                )
                t = nf.nets.MLP(
                    [self.dim, self.hidden_units] +
                    [self.hidden_units] * (self.hidden_layers - 1) +
                    [self.dim],
                    init_zeros=True,
                )
                # Alternate the mask
                if i % 2 == 0:
                    flows.append(nf.flows.MaskedAffineFlow(b, t, s))
                else:
                    flows.append(nf.flows.MaskedAffineFlow(1 - b, t, s))

        elif self.flow_type == 'maf':
            # Masked Autoregressive Flow
            for i in range(self.num_layers):
                flows.append(
                    nf.flows.MaskedAffineAutoregressive(
                        self.dim,
                        self.hidden_units,
                        num_blocks=self.hidden_layers,
                    )
                )
                # Add permutation between layers
                if i < self.num_layers - 1:
                    flows.append(nf.flows.Permute(self.dim, mode='swap'))

        elif self.flow_type == 'glow':
            # Glow-style flow with ActNorm and invertible 1x1 convolutions
            for i in range(self.num_layers):
                # Activation normalization
                flows.append(nf.flows.ActNorm(self.dim))
                # Invertible 1x1 convolution
                flows.append(nf.flows.InvertibleConv1x1(self.dim))
                # Affine coupling
                param_map = nf.nets.MLP(
                    [self.dim // 2, self.hidden_units] +
                    [self.hidden_units] * (self.hidden_layers - 1) +
                    [self.dim],
                    init_zeros=True,
                )
                flows.append(nf.flows.AffineCouplingBlock(param_map))

        elif self.flow_type == 'nsf':
            # Neural Spline Flow
            for i in range(self.num_layers):
                flows.append(
                    nf.flows.AutoregressiveRationalQuadraticSpline(
                        self.dim,
                        self.hidden_layers,
                        self.hidden_units,
                        num_bins=8,
                    )
                )
                # Add permutation between layers
                if i < self.num_layers - 1:
                    flows.append(nf.flows.Permute(self.dim, mode='swap'))

        elif self.flow_type == 'residual':
            # Residual flow
            for i in range(self.num_layers):
                net = nf.nets.LipschitzMLP(
                    [self.dim] + [self.hidden_units] * self.hidden_layers + [self.dim],
                    init_zeros=True,
                    lipschitz_const=0.9,
                )
                flows.append(nf.flows.Residual(net, reduce_memory=True))

        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")

        return flows

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward transformation (noise to data).

        This transforms latent samples from the base distribution to the data space.

        :param z: Latent noise tensor of shape (batch_size, dim)

        :returns: Transformed data tensor of shape (batch_size, dim)
        """
        return self.flow.forward(z)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse transformation (data to noise).

        This transforms data samples to the latent space of the base distribution.

        :param x: Input data tensor of shape (batch_size, dim)

        :returns: Latent tensor of shape (batch_size, dim)
        """
        return self.flow.inverse(x)

    def forward_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward transformation with log determinant computation.

        :param z: Latent noise tensor of shape (batch_size, dim)

        :returns: Tuple of (data_x, log_det_jacobian)
        """
        return self.flow.forward_and_log_det(z)

    def inverse_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the inverse transformation with log determinant computation.

        :param x: Input data tensor of shape (batch_size, dim)

        :returns: Tuple of (latent_z, log_det_jacobian)
        """
        return self.flow.inverse_and_log_det(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of data samples under the model.

        This method transforms samples from data space to noise space and
        computes the log probability using the change of variables formula.

        :param x: Data samples of shape (batch_size, dim)

        :returns: Log probabilities of shape (batch_size,)
        """
        return self.flow.log_prob(x)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Generate samples from the model.

        This method samples from the base distribution and transforms
        the samples to data space using the forward flow.

        :param num_samples: Number of samples to generate

        :returns: Generated samples of shape (num_samples, dim)
        """
        # normflows sample() returns a tuple of (samples, log_prob)
        # We only return the samples
        samples, _ = self.flow.sample(num_samples)
        return samples

    def training_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, ...], List],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step for PyTorch Lightning.

        Computes the negative log likelihood loss for the batch.

        :param batch: Batch of training data (can be tensor, tuple, or list)
        :param batch_idx: Index of the batch

        :returns: Loss value
        """
        # Handle both direct tensors and tuples/lists (e.g., from DataLoader)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        log_prob = self.log_prob(x)
        loss = -log_prob.mean()

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_log_prob', log_prob.mean(), prog_bar=True, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, ...], List],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step for PyTorch Lightning.

        :param batch: Batch of validation data (can be tensor, tuple, or list)
        :param batch_idx: Index of the batch

        :returns: Loss value
        """
        # Handle both direct tensors and tuples/lists
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        log_prob = self.log_prob(x)
        loss = -log_prob.mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_log_prob', log_prob.mean(), prog_bar=True)

        return loss

    def test_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, ...], List],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Test step for PyTorch Lightning.

        :param batch: Batch of test data (can be tensor, tuple, or list)
        :param batch_idx: Index of the batch

        :returns: Loss value
        """
        # Handle both direct tensors and tuples/lists
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        log_prob = self.log_prob(x)
        loss = -log_prob.mean()

        self.log('test_loss', loss)
        self.log('test_log_prob', log_prob.mean())

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        :returns: Adam optimizer with the specified learning rate
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
