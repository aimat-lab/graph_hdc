from typing import Generator
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import IterableDataset
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

# optimal transport
import ot

LOWER_LIMIT: int = 0
UPPER_LIMIT: int = 7
NUM_ISLANDS: int = 10
SEED: int = 2


class DiscreteDataset(IterableDataset):
    
    def __init__(
        self,
        lower_limit: int = 0,
        upper_limit: int = 5,
        num_islands: int = 10,
        seed: int = 0,
        epoch_size: int = 1000
    ) -> None:
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.num_islands = num_islands
        self.seed = seed
        self.epoch_size = epoch_size
        
        self.rng = np.random.default_rng(seed)
        self.islands = [
            self.rng.integers(lower_limit, upper_limit, size=(2,))
            for _ in range(num_islands)
        ]
    
    def __iter__(self) -> Generator[None, None, torch.Tensor]:
        
        for _ in range(self.epoch_size):
            
            x_center, y_center = self.rng.choice(self.islands)
            x = self.rng.uniform(x_center - 0.5, x_center + 0.5)
            y = self.rng.uniform(y_center - 0.5, y_center + 0.5)
            yield torch.tensor([x, y], dtype=torch.float32)


class FlowModel(pl.LightningModule):
    
    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        use_optimal_transport: bool = False,
    ) -> None:
        
        super().__init__()
        self.dim = dim
        self.learning_rate = learning_rate
        self.use_optimal_transport = use_optimal_transport
        
        self.lay_embedd_time = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.lay_embedd_data = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.lay_proj = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, dim),
        )
        
        # --- flow matching ---
        
        self.scheduler = CondOTScheduler()
        self.path = AffineProbPath(scheduler=self.scheduler)
        
        self.criterion = MixturePathGeneralizedKL(self.path)
        self.epsilon = 1e-3
        
    def ot_compute_cost_matrix(
        self,
        x_0: torch.Tensor, # (B, 2)
        x_1: torch.Tensor, # (B, 2)
    ) -> torch.Tensor:
        
        B, N = x_0.size()
        
        # efficient pairwise Euclidean distances (B, B)
        x_0_expanded = x_0.unsqueeze(1)  # (B, 1, N)
        x_1_expanded = x_1.unsqueeze(0)  # (1, B, N)
        dist_sq = torch.sum((x_0_expanded - x_1_expanded) ** 2, dim=2)  # (B, B)
        cost = torch.sqrt(dist_sq)

        return cost
    
    def ot_compute_coupling(
        self,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        
        batch_size = cost.size(0)
        cost_numpy = cost.detach().cpu().numpy()
        
        a = np.ones((batch_size,)) / batch_size
        b = np.ones((batch_size,)) / batch_size
        
        coupling = ot.sinkhorn(a, b, cost_numpy, reg=0.1)
        
        return torch.tensor(coupling)
    
    def ot_apply_coupling(
        self,
        x_0: torch.Tensor, # (B, 2)
        x_1: torch.Tensor, # (B, 2)
        coupling: torch.Tensor, # (B, B)
    ) -> torch.Tensor:
        
        batch_size = x_0.size(0)
        
        # Sample from the coupling matrix
        coupling_flat = coupling.flatten()
        coupling_probs = coupling_flat / coupling_flat.sum()
        
        # Sample B pairs according to coupling probabilities
        indices = torch.multinomial(coupling_probs, batch_size, replacement=True)
        i_indices = indices // batch_size
        j_indices = indices % batch_size
        
        x0_paired = x_0[i_indices]
        x1_paired = x_1[j_indices]
    
        return x0_paired, x1_paired
        
    # This method is supposed to take a current guess/interpolation of the data point x 
    # at the time t as an input and is supposed to predict the direction of moving the 
    # point x to the target data point that is supposed to be reached at time t=1.
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        
        time_embedding = self.lay_embedd_time(t.unsqueeze(-1))
        data_embedding = self.lay_embedd_data(x)
        
        combined = torch.cat([time_embedding, data_embedding], dim=-1)
        flow = self.lay_proj(combined)
        
        return flow
    
    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.rand((batch_size, self.dim), device=self.device) * (UPPER_LIMIT - LOWER_LIMIT) + LOWER_LIMIT
    
    def training_step(
        self,
        batch: torch.Tensor, 
        batch_idx: int
    ) -> torch.Tensor:
        
        batch_size = batch.size(0)
        
        # sample time
        t = torch.rand((batch_size, ), device=batch.device) * (1.0 - self.epsilon)
        
        # sample prior
        x_0 = self.sample_noise(batch_size)
        x_1 = batch
        
        if self.use_optimal_transport:
            cost = self.ot_compute_cost_matrix(x_0, x_1)
            coupling = self.ot_compute_coupling(cost)
            x_0, x_1 = self.ot_apply_coupling(x_0, x_1, coupling)
        
        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        pred = self.forward(path_sample.x_t, path_sample.t)
        loss = torch.pow(pred - path_sample.dx_t, 2).mean()

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    


class FlowModelWrapper(ModelWrapper):
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        
        batch_size = x.size(0)
        return self.model(x, t.expand(batch_size), **kwargs)
    
    # --- inference ---
    
    def sample(
        self, 
        batch_size: int = 10, 
        time_steps: int = 50
    ) -> torch.Tensor:
        
        solver = ODESolver(velocity_model=self)
        
        x_0 = self.model.sample_noise(batch_size)
        
        time_grid = torch.linspace(0.0, 1.0, time_steps)
        time_grid = time_grid.to(self.model.device)
        print('time grid', time_grid)
        samples = solver.sample(
            x_init=x_0,
            time_grid=time_grid, 
            method='midpoint', 
            step_size=0.01, 
            return_intermediates=False,
        )
        
        return samples


experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    e.log_parameters()
    
    # --- initialize dataset ---
    # At first we need to initialize the mock dataset that we are using. In this case
    # the dataset will be a discrete distribution of points in a 2D space. Only specific
    # areas in that space will have points while most of them will not at all (discrete).
    # The dataset class itself is an infinitely streaming IterableDataset
    e.log('initializing the dataset...')
    dataset = DiscreteDataset(
        lower_limit=e.LOWER_LIMIT,
        upper_limit=e.UPPER_LIMIT,
        num_islands=e.NUM_ISLANDS,
        seed=e.SEED,
        epoch_size=100_000,
    )
    
    e.log('visualizing the data distribution by sampling points from the dataset...')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # collect samples from the iterable dataset into a single array
    samples = []
    for i, sample in enumerate(dataset):
        samples.append(sample.numpy())
    samples = np.vstack(samples)  # shape (N, 2)

    # plot a 2D histogram / density instead of plotting every point
    h = ax.hist2d(
        samples[:, 0],
        samples[:, 1],
        bins=100,
        range=[
            [e.LOWER_LIMIT - 1, e.UPPER_LIMIT + 1],
            [e.LOWER_LIMIT - 1, e.UPPER_LIMIT + 1]
        ],
        cmap="viridis",
    )
    ax.set_ylim(e.LOWER_LIMIT - 1, e.UPPER_LIMIT + 1)
    ax.set_xlim(e.LOWER_LIMIT - 1, e.UPPER_LIMIT + 1)
    fig.colorbar(h[3], ax=ax, label="counts")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Data distribution (2D histogram)")
    ax.set_aspect("equal", adjustable="box")
        
    e.commit_fig('data_distribution.png', fig)
    
    # --- Construct model ---
    model = FlowModel(
        dim=2,
        hidden_dim=256,
        learning_rate=1e-3,
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=4,
        prefetch_factor=2,
    )
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices='auto',
        enable_progress_bar=True,
        logger=False,
        enable_model_summary=True,
        limit_train_batches=100,
    )
    
    # --- Model Training ---
    
    e.log('starting training...')
    trainer.fit(model, data_loader)
    model.eval()
    
    # --- Model Evaluation ---
    
    e.log('starting evaluation...')
    
    num_samples = 2000
    
    wrapped = FlowModelWrapper(model)
    samples = wrapped.sample(batch_size=num_samples, time_steps=100)
    samples = samples.cpu().numpy()
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    for sample in samples:
        x, y = sample
        ax.scatter(x, y, color='blue', alpha=0.5)

    e.commit_fig('generated_samples.png', fig)
    

experiment.run_if_main()
