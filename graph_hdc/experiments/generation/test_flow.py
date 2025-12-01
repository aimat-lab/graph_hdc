"""
Binary Graph Edge Reconstruction with Conditioning using PyTorch Lightning
Simplified from GGFlow concepts for binary adjacency matrix reconstruction

This implementation focuses on binary edge prediction (exists/doesn't exist)
with additional conditioning information per sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from dataclasses import dataclass

# Core flow_matching imports
from flow_matching.path.scheduler import PolynomialConvexScheduler
import ot  # For optimal transport


@dataclass
class BinaryEdgeConfig:
    """Configuration for binary edge reconstruction with conditioning"""
    # Graph properties  
    max_nodes: int = 50
    node_feature_dim: int = 64  # Dimension of input node features
    condition_dim: int = 32     # Dimension of conditioning vector
    
    # Model architecture
    hidden_dim: int = 256
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Flow matching parameters
    num_train_timesteps: int = 1000
    scheduler_power: float = 2.0  # κ_t = t^2
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Sampling
    num_inference_steps: int = 50
    
    # Optimal transport
    ot_reg: float = 0.05
    use_minibatch_ot: bool = True
    
    # Training details
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    validation_check_interval: float = 0.25


class BinaryGraphData:
    """
    Represents graph data with fixed nodes, binary edges, and conditioning
    Much simpler than the multi-type edge case
    """
    
    def __init__(self, node_features: torch.Tensor, adjacency: torch.Tensor, 
                 condition: torch.Tensor, num_real_nodes: Optional[torch.Tensor] = None):
        """
        Args:
            node_features: [batch_size, max_nodes, node_feature_dim] - fixed node features
            adjacency: [batch_size, max_nodes, max_nodes] - binary adjacency matrix (0/1)
            condition: [batch_size, condition_dim] - conditioning vector per sample
            num_real_nodes: [batch_size] - actual number of nodes per sample
        """
        self.node_features = node_features
        self.adjacency = adjacency  # Binary: 1 for edge exists, 0 for no edge
        self.condition = condition
        self.num_real_nodes = num_real_nodes
        
    def get_node_mask(self) -> torch.Tensor:
        """Create mask for real nodes vs padding"""
        if self.num_real_nodes is not None:
            batch_size, max_nodes = self.node_features.shape[:2]
            node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
            for i, num_nodes in enumerate(self.num_real_nodes):
                node_mask[i, :num_nodes] = True
            return node_mask.float()
        else:
            # Fallback: assume non-zero node features indicate real nodes
            return (self.node_features.abs().sum(dim=-1) > 1e-6).float()
    
    def get_edge_mask(self) -> torch.Tensor:
        """Create mask for valid edges (between real nodes)"""
        node_mask = self.get_node_mask()  # [batch_size, max_nodes]
        
        # Edge is valid if both endpoints are real nodes
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)  # [batch_size, max_nodes, max_nodes]
        
        # Remove self-loops if desired
        batch_size, max_nodes = node_mask.shape
        eye = torch.eye(max_nodes).unsqueeze(0).expand(batch_size, -1, -1)
        edge_mask = edge_mask * (1 - eye)  # Set diagonal to 0
        
        return edge_mask


class ConditionalTriangleAttention(nn.Module):
    """
    Triangle attention for binary edges with conditioning support
    This is the core mechanism for edge-to-edge communication
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, condition_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Standard triangle attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Conditioning integration - condition affects attention computation
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.condition_gate = nn.Linear(condition_dim, hidden_dim)
        
        # Triangle-specific components
        self.triangle_bias = nn.Linear(hidden_dim + condition_dim, num_heads)
        self.triangle_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, edge_features: torch.Tensor, edge_mask: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        Apply conditional triangle attention to edge features
        Args:
            edge_features: [batch_size, max_nodes, max_nodes, hidden_dim]
            edge_mask: [batch_size, max_nodes, max_nodes] - mask for valid edges  
            condition: [batch_size, condition_dim] - conditioning vector
        """
        batch_size, max_nodes, _, hidden_dim = edge_features.shape
        
        # Project conditioning vector
        cond_proj = self.condition_proj(condition)  # [batch_size, hidden_dim]
        cond_gate = torch.sigmoid(self.condition_gate(condition))
        
        # Add conditioning to edge features
        cond_expanded = cond_proj.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, hidden_dim]
        conditioned_features = edge_features * cond_gate.unsqueeze(1).unsqueeze(2) + cond_expanded
        
        # Compute Q, K, V with conditioning
        Q = self.q_proj(conditioned_features).view(batch_size, max_nodes, max_nodes, 
                                                  self.num_heads, self.head_dim)
        K = self.k_proj(conditioned_features).view(batch_size, max_nodes, max_nodes, 
                                                  self.num_heads, self.head_dim)
        V = self.v_proj(conditioned_features).view(batch_size, max_nodes, max_nodes, 
                                                  self.num_heads, self.head_dim)
        
        # Triangle attention computation
        # Each edge attends to all other edges, weighted by relevance
        scores = torch.einsum('bijhd,bklhd->bijklh', Q, K) / np.sqrt(self.head_dim)
        
        # Add learnable bias that depends on both edge features and condition
        bias_input = torch.cat([
            conditioned_features, 
            condition.unsqueeze(1).unsqueeze(2).expand(-1, max_nodes, max_nodes, -1)
        ], dim=-1)
        bias = self.triangle_bias(bias_input)  # [batch_size, max_nodes, max_nodes, num_heads]
        scores = scores + bias.unsqueeze(3).unsqueeze(4)
        
        # Apply edge mask to prevent attention to invalid edges
        mask_expanded = edge_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)  # For i,j positions
        mask_expanded = mask_expanded * edge_mask.unsqueeze(3).unsqueeze(4).unsqueeze(-1)  # For k,l positions
        
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Apply attention weights
        attn_weights = F.softmax(scores, dim=4)  # Attend over k,l dimensions
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.einsum('bijklh,bklhd->bijhd', attn_weights, V)
        attended = attended.reshape(batch_size, max_nodes, max_nodes, hidden_dim)
        
        # Final gating and output projection
        gate = torch.sigmoid(self.triangle_gate(attended))
        output = attended * gate
        
        return self.output_proj(output)


class ConditionalNodeToEdgeAttention(nn.Module):
    """
    Lets edges attend to node features, with conditioning support
    This provides context about what types of nodes are being potentially connected
    """
    
    def __init__(self, node_dim: int, edge_dim: int, condition_dim: int, 
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = edge_dim // num_heads
        
        # Node feature processing with conditioning
        self.node_proj = nn.Linear(node_dim + condition_dim, edge_dim)
        self.node_to_key = nn.Linear(edge_dim, edge_dim)
        self.node_to_value = nn.Linear(edge_dim, edge_dim)
        
        # Edge query processing
        self.edge_to_query = nn.Linear(edge_dim, edge_dim)
        
        self.output_proj = nn.Linear(edge_dim, edge_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, edge_features: torch.Tensor, node_features: torch.Tensor,
                edge_mask: torch.Tensor, node_mask: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        Let edges attend to conditioned node features
        """
        batch_size, max_nodes, _, edge_dim = edge_features.shape
        
        # Condition node features - each node gets the same global conditioning
        cond_expanded = condition.unsqueeze(1).expand(-1, max_nodes, -1)  # [batch_size, max_nodes, condition_dim]
        conditioned_nodes = torch.cat([node_features, cond_expanded], dim=-1)
        
        # Project conditioned node features
        node_projected = self.node_proj(conditioned_nodes)  # [batch_size, max_nodes, edge_dim]
        node_keys = self.node_to_key(node_projected)
        node_values = self.node_to_value(node_projected)
        
        # Edge queries
        edge_queries = self.edge_to_query(edge_features)
        
        # For each edge (i,j), attend to both endpoint nodes i and j
        # This gives edges information about what they're connecting
        
        # Attend to source nodes (i)
        scores_i = torch.einsum('bijd,bid->bij', edge_queries, node_keys) / np.sqrt(self.head_dim)
        scores_i = scores_i.masked_fill(node_mask.unsqueeze(1) == 0, float('-inf'))
        attn_i = F.softmax(scores_i, dim=-1)
        attn_i = self.dropout(attn_i)
        attended_i = torch.einsum('bij,bid->bijd', attn_i, node_values)
        
        # Attend to target nodes (j)
        scores_j = torch.einsum('bijd,bjd->bij', edge_queries, node_keys) / np.sqrt(self.head_dim)
        scores_j = scores_j.masked_fill(node_mask.unsqueeze(2) == 0, float('-inf'))
        attn_j = F.softmax(scores_j, dim=-1)
        attn_j = self.dropout(attn_j)
        attended_j = torch.einsum('bij,bjd->bijd', attn_j, node_values)
        
        # Combine information from both endpoints
        node_context = attended_i + attended_j
        
        # Apply edge mask and output projection
        node_context = node_context * edge_mask.unsqueeze(-1)
        
        return self.output_proj(node_context)


class BinaryEdgeReconstructionLayer(nn.Module):
    """Single transformer layer for binary edge reconstruction with conditioning"""
    
    def __init__(self, config: BinaryEdgeConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        condition_dim = config.condition_dim
        
        # Triangle attention for edge-to-edge communication
        self.triangle_attention = ConditionalTriangleAttention(
            hidden_dim, config.num_attention_heads, condition_dim, config.dropout_rate
        )
        
        # Node-to-edge attention for incorporating node context
        self.node_to_edge_attention = ConditionalNodeToEdgeAttention(
            hidden_dim, hidden_dim, condition_dim, config.num_attention_heads, config.dropout_rate
        )
        
        # Layer normalization and feed-forward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, edge_features: torch.Tensor, node_features: torch.Tensor,
                edge_mask: torch.Tensor, node_mask: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        
        # Triangle attention: edges attend to other edges with conditioning
        triangle_out = self.triangle_attention(edge_features, edge_mask, condition)
        edge_features = self.norm1(edge_features + triangle_out)
        
        # Node context: edges attend to their conditioned endpoint nodes
        node_context = self.node_to_edge_attention(
            edge_features, node_features, edge_mask, node_mask, condition
        )
        edge_features = self.norm2(edge_features + node_context)
        
        # Feed-forward network
        ffn_out = self.ffn(edge_features)
        edge_features = self.norm3(edge_features + ffn_out)
        
        return edge_features


class BinaryEdgeFlowModel(pl.LightningModule):
    """
    PyTorch Lightning module for binary edge reconstruction using flow matching
    This is the main model that orchestrates training, validation, and sampling
    """
    
    def __init__(self, config: BinaryEdgeConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Time and condition embedding for flow matching
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Condition embedding - transforms condition vector to hidden space
        self.condition_embedding = nn.Sequential(
            nn.Linear(config.condition_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Project input node features to hidden dimension
        self.node_proj = nn.Linear(config.node_feature_dim, config.hidden_dim)
        
        # Initial edge embedding (for current binary adjacency state)
        self.edge_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_dim),  # Single value: edge exists or not
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Transformer layers for edge processing
        self.edge_layers = nn.ModuleList([
            BinaryEdgeReconstructionLayer(config) 
            for _ in range(config.num_transformer_layers)
        ])
        
        # Output projection to binary edge probabilities
        self.edge_output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1)  # Single logit for binary classification
        )
        
        # Flow matching scheduler
        self.scheduler = PolynomialConvexScheduler(n=config.scheduler_power)
        
        # Metrics for tracking performance
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        
        # For tracking OT coupling quality
        self.register_buffer('ot_cost_history', torch.zeros(100))  # Rolling history of OT costs
        self.ot_step_counter = 0
        
    def forward(self, graph_data: BinaryGraphData, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict binary edge probabilities given current state and time
        Args:
            graph_data: BinaryGraphData with fixed nodes, current edges, and condition
            t: time parameter [batch_size]
        Returns:
            edge_logits: [batch_size, max_nodes, max_nodes] - logits for edge existence
        """
        batch_size = graph_data.node_features.shape[0]
        
        # Get masks for valid nodes and edges
        node_mask = graph_data.get_node_mask()  # [batch_size, max_nodes]
        edge_mask = graph_data.get_edge_mask()  # [batch_size, max_nodes, max_nodes]
        
        # Time embedding
        t_embed = self.time_embedding(t.unsqueeze(-1))  # [batch_size, hidden_dim]
        
        # Condition embedding - this modulates the entire generation process
        cond_embed = self.condition_embedding(graph_data.condition)  # [batch_size, hidden_dim]
        
        # Combine time and condition - both affect the generation process
        temporal_context = t_embed + cond_embed  # [batch_size, hidden_dim]
        
        # Process node features (fixed throughout the flow)
        node_features = self.node_proj(graph_data.node_features)  # [batch_size, max_nodes, hidden_dim]
        
        # Add temporal context to node features
        node_features = node_features + temporal_context.unsqueeze(1)
        
        # Process current edge state (binary values)
        edge_values = graph_data.adjacency.unsqueeze(-1).float()  # [batch_size, max_nodes, max_nodes, 1]
        edge_features = self.edge_embedding(edge_values)  # [batch_size, max_nodes, max_nodes, hidden_dim]
        
        # Add temporal context to edge features
        edge_features = edge_features + temporal_context.unsqueeze(1).unsqueeze(2)
        
        # Apply transformer layers with conditioning
        for layer in self.edge_layers:
            edge_features = layer(
                edge_features, node_features, edge_mask, node_mask, graph_data.condition
            )
        
        # Output projection to binary logits
        edge_logits = self.edge_output(edge_features).squeeze(-1)  # [batch_size, max_nodes, max_nodes]
        
        # Ensure symmetry for undirected graphs
        edge_logits = (edge_logits + edge_logits.transpose(-1, -2)) / 2
        
        # Apply edge mask to set invalid edges to zero probability
        edge_logits = edge_logits * edge_mask
        
        return edge_logits
    
    def create_binary_edge_prior(self, batch_size: int, max_nodes: int, 
                                condition: torch.Tensor) -> torch.Tensor:
        """
        Create sparse prior distribution for binary edges
        The sparsity can be influenced by the conditioning vector
        """
        device = self.device
        
        # Base sparsity level - most graphs are sparse
        base_sparsity = 0.05  # 5% of possible edges exist
        
        # Let conditioning influence sparsity (simple linear relationship)
        # This is a design choice - you might want more sophisticated conditioning
        sparsity_modulation = torch.sigmoid(condition.mean(dim=-1, keepdim=True))  # [batch_size, 1]
        adjusted_sparsity = base_sparsity * (0.5 + sparsity_modulation.cpu())  # Between 2.5% and 7.5%
        
        # Sample binary adjacency matrices
        priors = []
        for i in range(batch_size):
            # Sample edge probabilities
            edge_probs = torch.full((max_nodes, max_nodes), adjusted_sparsity[i].item())
            
            # Sample binary adjacency
            binary_adj = torch.bernoulli(edge_probs)
            
            # Make symmetric (undirected graph)
            binary_adj = torch.triu(binary_adj, diagonal=1)  # Upper triangle only
            binary_adj = binary_adj + binary_adj.T  # Make symmetric
            
            priors.append(binary_adj)
        
        return torch.stack(priors).to(device)
    
    def hamming_distance_batch(self, adj1: torch.Tensor, adj2: torch.Tensor,
                              edge_mask1: torch.Tensor, edge_mask2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Hamming distances between batches of binary adjacency matrices
        This is the cost function for optimal transport

        Args:
            adj1: [batch_size, max_nodes, max_nodes] - first batch of adjacency matrices
            adj2: [batch_size, max_nodes, max_nodes] - second batch of adjacency matrices
            edge_mask1: [batch_size, max_nodes, max_nodes] - mask for valid edges in adj1
            edge_mask2: [batch_size, max_nodes, max_nodes] - mask for valid edges in adj2
        Returns:
            cost_matrix: [batch_size, batch_size] - pairwise Hamming distances
        """
        batch_size = adj1.shape[0]
        cost_matrix = torch.zeros(batch_size, batch_size, device=adj1.device)

        for i in range(batch_size):
            for j in range(batch_size):
                # Use intersection of both masks to only compare valid edges
                combined_mask = edge_mask1[i] * edge_mask2[j]

                # Count different edges between adjacency matrices i and j
                # Only count differences where both positions are valid
                diff = ((adj1[i] != adj2[j]) * combined_mask).sum()
                cost_matrix[i, j] = diff

        return cost_matrix
    
    def compute_ot_coupling(self, prior_adj: torch.Tensor, target_adj: torch.Tensor,
                           prior_mask: torch.Tensor, target_mask: torch.Tensor,
                           device: torch.device) -> torch.Tensor:
        """
        Compute optimal transport coupling matrix using Sinkhorn algorithm
        This finds the best way to pair noise samples with target samples

        Args:
            prior_adj: [batch_size, max_nodes, max_nodes] - noise adjacency matrices
            target_adj: [batch_size, max_nodes, max_nodes] - target adjacency matrices
            prior_mask: [batch_size, max_nodes, max_nodes] - mask for valid edges in prior
            target_mask: [batch_size, max_nodes, max_nodes] - mask for valid edges in target
        Returns:
            coupling: [batch_size, batch_size] - optimal transport coupling matrix
        """
        batch_size = prior_adj.shape[0]

        # Compute cost matrix using Hamming distance
        cost_matrix = self.hamming_distance_batch(prior_adj, target_adj, prior_mask, target_mask)
        
        # Uniform marginal distributions (each sample has equal weight)
        a = torch.ones(batch_size, device=device) / batch_size
        b = torch.ones(batch_size, device=device) / batch_size
        
        # Use Python Optimal Transport library for Sinkhorn algorithm
        cost_cpu = cost_matrix.cpu().numpy()
        a_cpu = a.cpu().numpy()
        b_cpu = b.cpu().numpy()

        # Add small epsilon for numerical stability
        cost_cpu = cost_cpu + 1e-8

        try:
            # Sinkhorn algorithm with entropy regularization
            coupling_cpu = ot.sinkhorn(a_cpu, b_cpu, cost_cpu, reg=self.config.ot_reg, numItermax=1000, stopThr=1e-6)
            coupling = torch.from_numpy(coupling_cpu).float().to(device)

            # Check for numerical issues
            if torch.isnan(coupling).any() or torch.isinf(coupling).any():
                print("Warning: NaN or Inf in OT coupling, using identity coupling")
                coupling = torch.eye(batch_size, device=device)
        except Exception as e:
            print(f"Warning: OT failed with error {e}, using identity coupling")
            coupling = torch.eye(batch_size, device=device)
        
        # Track OT cost for monitoring
        ot_cost = (coupling * cost_matrix).sum()
        self.log_ot_cost(ot_cost)
        
        return coupling
    
    def apply_ot_coupling(self, prior_adj: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        """
        Apply optimal transport coupling to rearrange prior samples
        This creates better noise-target pairings for more efficient learning
        
        Args:
            prior_adj: [batch_size, max_nodes, max_nodes] - original prior samples
            coupling: [batch_size, batch_size] - OT coupling matrix
        Returns:
            rearranged_prior: [batch_size, max_nodes, max_nodes] - rearranged prior samples
        """
        batch_size = prior_adj.shape[0]
        
        # Sample indices according to coupling probabilities
        # For each target, sample a prior according to the coupling distribution
        rearranged_indices = []
        for i in range(batch_size):
            # Get coupling probabilities for target i
            probs = coupling[:, i]

            # Check for invalid probabilities
            if probs.sum() <= 0 or torch.isnan(probs).any() or torch.isinf(probs).any():
                # Fallback to uniform sampling
                sampled_idx = torch.randint(0, batch_size, (1,)).item()
            else:
                # Normalize probabilities to ensure they sum to 1
                probs = probs / probs.sum()
                # Sample prior index according to these probabilities
                sampled_idx = torch.multinomial(probs, 1).item()
            rearranged_indices.append(sampled_idx)
        
        # Rearrange prior adjacency matrices
        rearranged_prior = prior_adj[rearranged_indices]
        
        return rearranged_prior
    
    def log_ot_cost(self, cost: torch.Tensor):
        """Track optimal transport cost over time for monitoring training dynamics"""
        idx = self.ot_step_counter % self.ot_cost_history.shape[0]
        self.ot_cost_history[idx] = cost.detach()
        self.ot_step_counter += 1
        
        # Log average OT cost periodically
        if self.ot_step_counter % 100 == 0:
            avg_cost = self.ot_cost_history.mean()
            self.log('train/ot_cost', avg_cost, on_step=True)
    
    def flow_matching_loss(self, batch: BinaryGraphData) -> torch.Tensor:
        """
        Compute flow matching loss for binary edge reconstruction with optimal transport
        This properly implements the OT coupling from the GGFlow paper
        """
        batch_size = batch.node_features.shape[0]
        device = self.device
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=device)
        
        # Get target binary adjacency matrices
        target_adj = batch.adjacency.float().to(device)  # [batch_size, max_nodes, max_nodes]
        
        # Create edge prior (sparse random adjacency matrices)
        prior_adj = self.create_binary_edge_prior(
            batch_size, self.config.max_nodes, batch.condition
        )
        
        # Apply optimal transport coupling to find better noise-target pairings
        if self.config.use_minibatch_ot and batch_size > 1:
            # Create edge masks for both prior and target
            # For prior: uniform mask since we generated full adjacency matrices
            prior_mask = torch.ones_like(prior_adj)
            # For target: get proper edge mask from batch
            target_graph_data = BinaryGraphData(
                node_features=batch.node_features,
                adjacency=target_adj,
                condition=batch.condition,
                num_real_nodes=batch.num_real_nodes
            )
            target_mask = target_graph_data.get_edge_mask()

            # Compute optimal transport coupling within this minibatch
            coupling_matrix = self.compute_ot_coupling(
                prior_adj, target_adj, prior_mask, target_mask, device
            )

            # Rearrange prior samples according to optimal coupling
            # The coupling matrix gives us transport probabilities between noise and targets
            prior_adj = self.apply_ot_coupling(prior_adj, coupling_matrix)
        
        # Create probability paths using linear interpolation
        # For binary case, this interpolates between Bernoulli parameters
        interpolated_probs = (1 - t.view(-1, 1, 1)) * prior_adj + t.view(-1, 1, 1) * target_adj
        
        # Sample from interpolated Bernoulli distributions
        interpolated_adj = torch.bernoulli(interpolated_probs)
        
        # Create graph data for model input
        graph_t = BinaryGraphData(
            node_features=batch.node_features.to(device),
            adjacency=interpolated_adj,
            condition=batch.condition.to(device),
            num_real_nodes=batch.num_real_nodes.to(device) if batch.num_real_nodes is not None else None
        )
        
        # Model prediction
        edge_logits = self(graph_t, t)
        
        # Binary cross-entropy loss
        # We want to predict the target adjacency matrix
        edge_mask = graph_t.get_edge_mask()
        
        # Only compute loss on valid edges
        loss = F.binary_cross_entropy_with_logits(
            edge_logits[edge_mask > 0], 
            target_adj[edge_mask > 0],
            reduction='mean'
        )
        
        return loss
    
    def training_step(self, batch: BinaryGraphData, batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning"""
        loss = self.flow_matching_loss(batch)

        # Log metrics with explicit batch size
        batch_size = batch.node_features.shape[0]
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, batch_size=batch_size)

        return loss
    
    def validation_step(self, batch: BinaryGraphData, batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning"""
        loss = self.flow_matching_loss(batch)
        
        # Skip sampling during sanity check to avoid issues with untrained model
        if not self.trainer.sanity_checking:
            # Also evaluate reconstruction quality by sampling
            with torch.no_grad():
                try:
                    # Sample edges using the current model
                    sampled_adj = self.sample_edges(
                        batch.node_features[:4],  # Just first 4 samples to save time
                        batch.condition[:4],
                        batch.num_real_nodes[:4] if batch.num_real_nodes is not None else None
                    )

                    # Compute metrics against ground truth
                    target_adj = batch.adjacency[:4].float()
                    edge_mask = BinaryGraphData(
                        batch.node_features[:4], target_adj, batch.condition[:4], batch.num_real_nodes[:4]
                    ).get_edge_mask()

                    # Only evaluate on valid edges
                    valid_indices = edge_mask > 0
                    if valid_indices.sum() > 0:
                        self.val_acc.update(sampled_adj[valid_indices], target_adj[valid_indices].int())
                        self.val_f1.update(sampled_adj[valid_indices], target_adj[valid_indices].int())
                        self.val_auroc.update(sampled_adj[valid_indices].float(), target_adj[valid_indices].int())
                except Exception as e:
                    print(f"Warning: Sampling failed during validation: {e}")
                    # Continue without sampling metrics
        
        # Log validation metrics with explicit batch size
        batch_size = batch.node_features.shape[0]
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Only log metrics if not in sanity check (to avoid empty metric issues)
        if not self.trainer.sanity_checking:
            self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)  # Better for transformer architectures
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                progress = (step - self.config.warmup_steps) / (self.trainer.estimated_stepping_batches - self.config.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    @torch.no_grad()
    def sample_edges(self, node_features: torch.Tensor, condition: torch.Tensor,
                    num_real_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate binary adjacency matrix given fixed node features and conditioning
        This is your main inference function for generating graph structures
        """
        self.eval()
        batch_size, max_nodes = node_features.shape[:2]
        device = self.device
        
        # Start from sparse edge prior
        current_adj = self.create_binary_edge_prior(batch_size, max_nodes, condition)
        
        # Integration steps using Euler method
        dt = 1.0 / self.config.num_inference_steps
        
        for step in range(self.config.num_inference_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            
            # Create current graph state
            graph_t = BinaryGraphData(
                node_features=node_features.to(device),
                adjacency=current_adj,
                condition=condition.to(device),
                num_real_nodes=num_real_nodes.to(device) if num_real_nodes is not None else None
            )
            
            # Model prediction
            edge_logits = self(graph_t, t)
            
            # Convert to probabilities and sample
            edge_probs = torch.sigmoid(edge_logits)

            # Clamp probabilities to valid range for numerical stability
            edge_probs = torch.clamp(edge_probs, min=1e-6, max=1-1e-6)

            # For the final steps, use more deterministic sampling
            if step > self.config.num_inference_steps * 0.8:
                # Use probabilities directly for final refinement
                current_adj = (edge_probs > 0.5).float()
            else:
                # Stochastic sampling in early steps
                current_adj = torch.bernoulli(edge_probs)
        
        self.train()
        return current_adj.int()


class GraphDataset(Dataset):
    """
    PyTorch Dataset for graph data with binary edges and conditioning
    This handles loading and preprocessing of your graph data
    """
    
    def __init__(self, node_features_list: List[torch.Tensor], 
                 adjacency_list: List[torch.Tensor],
                 condition_list: List[torch.Tensor],
                 num_nodes_list: Optional[List[int]] = None):
        """
        Args:
            node_features_list: List of [num_nodes, node_feature_dim] tensors
            adjacency_list: List of [num_nodes, num_nodes] binary adjacency matrices  
            condition_list: List of [condition_dim] conditioning vectors
            num_nodes_list: List of actual number of nodes per graph (for padding)
        """
        self.node_features_list = node_features_list
        self.adjacency_list = adjacency_list
        self.condition_list = condition_list
        self.num_nodes_list = num_nodes_list or [adj.shape[0] for adj in adjacency_list]
        
        # Determine max_nodes for padding - should be consistent with config
        self.max_nodes = max(adj.shape[0] for adj in adjacency_list)
        
    def __len__(self):
        return len(self.node_features_list)
    
    def __getitem__(self, idx: int) -> BinaryGraphData:
        # Get raw data - already padded to max_nodes in create_dummy_dataset
        node_features = self.node_features_list[idx]
        adjacency = self.adjacency_list[idx]
        condition = self.condition_list[idx]
        num_real_nodes = self.num_nodes_list[idx]

        return BinaryGraphData(
            node_features=node_features.unsqueeze(0),  # Add batch dimension
            adjacency=adjacency.unsqueeze(0),
            condition=condition.unsqueeze(0),
            num_real_nodes=torch.tensor([num_real_nodes])
        )


def create_dummy_dataset(num_samples: int = 1000, max_nodes: int = 20,
                        node_feature_dim: int = 64, condition_dim: int = 32) -> GraphDataset:
    """
    Create a dummy dataset for testing the implementation
    In practice, you'd replace this with your actual data loading
    """
    node_features_list = []
    adjacency_list = []
    condition_list = []
    num_nodes_list = []

    for _ in range(num_samples):
        # Random number of nodes - but always create the full matrix
        num_real_nodes = torch.randint(5, max_nodes + 1, (1,)).item()

        # Create padded node features - fill only first num_real_nodes rows
        node_features = torch.zeros(max_nodes, node_feature_dim)
        node_features[:num_real_nodes] = torch.randn(num_real_nodes, node_feature_dim)

        # Create padded adjacency matrix - only connect real nodes
        adjacency = torch.zeros(max_nodes, max_nodes)

        # Add some structure - connect consecutive nodes and add some random edges
        for i in range(num_real_nodes - 1):
            if torch.rand(1) > 0.3:  # 70% chance to connect to next node
                adjacency[i, i+1] = 1
                adjacency[i+1, i] = 1

        # Add some random edges within the real nodes
        num_random_edges = torch.randint(0, min(num_real_nodes, 5), (1,)).item()
        for _ in range(num_random_edges):
            i, j = torch.randint(0, num_real_nodes, (2,))
            if i != j:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

        # Random conditioning vector
        condition = torch.randn(condition_dim)

        node_features_list.append(node_features)
        adjacency_list.append(adjacency)
        condition_list.append(condition)
        num_nodes_list.append(num_real_nodes)

    return GraphDataset(node_features_list, adjacency_list, condition_list, num_nodes_list)


def custom_collate_fn(batch: List[BinaryGraphData]) -> BinaryGraphData:
    """Custom collate function to handle BinaryGraphData batching"""
    node_features = torch.cat([item.node_features for item in batch], dim=0)
    adjacency = torch.cat([item.adjacency for item in batch], dim=0)
    condition = torch.cat([item.condition for item in batch], dim=0)
    num_real_nodes = torch.cat([item.num_real_nodes for item in batch], dim=0)
    
    return BinaryGraphData(node_features, adjacency, condition, num_real_nodes)


def train_model():
    """
    Main training function using PyTorch Lightning
    This demonstrates how to set up and run the training process
    """
    
    # Configuration
    config = BinaryEdgeConfig(
        max_nodes=20,
        node_feature_dim=64,
        condition_dim=32,
        batch_size=16,
        max_epochs=50,
        num_inference_steps=25
    )
    
    # Create datasets
    print("Creating dummy dataset...")
    full_dataset = create_dummy_dataset(
        num_samples=2000,
        max_nodes=config.max_nodes,
        node_feature_dim=config.node_feature_dim,
        condition_dim=config.condition_dim
    )
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = BinaryEdgeFlowModel(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='binary-edge-flow-{epoch:02d}-{val/loss:.3f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode='min'
    )
    
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='cpu',  # Force CPU usage
        devices=1,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=config.validation_check_interval,
        log_every_n_steps=50
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test sampling
    print("\nTesting edge reconstruction...")
    model.eval()
    
    # Get a test batch
    test_batch = next(iter(val_loader))
    
    with torch.no_grad():
        # Sample edges for first few samples
        sampled_edges = model.sample_edges(
            test_batch.node_features[:4], 
            test_batch.condition[:4],
            test_batch.num_real_nodes[:4]
        )
        
        print(f"Generated adjacency matrices shape: {sampled_edges.shape}")
        print(f"Average edge density: {sampled_edges.float().mean():.3f}")
        
        # Compare with ground truth
        target_edges = test_batch.adjacency[:4]
        print(f"Target edge density: {target_edges.float().mean():.3f}")


if __name__ == "__main__":
    train_model()