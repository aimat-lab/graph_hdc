# SHAP Attribution Extension for HDC Encoder

## Motivation

The HDC encoder produces a graph embedding as a plain sum of per-node, per-layer
hypervectors:

```
g = Σ_k Σ_i  h_i^(k)
```

where `h_i^(k)` is the hypervector for node `i` at message-passing layer `k`
(encoding the k-hop neighborhood around that node). Each of these terms
represents a unique symbol in the high-dimensional space created by the binding
operation.

This additive structure is ideal for Shapley-value-based attribution: we treat
each `h_i^(k)` as a **player** in a cooperative game. A coalition is formed by
summing only the included players, and the downstream ML model evaluates the
resulting partial embedding. No marginalization is needed — absence simply means
not adding the term.

## Validation Strategy

CLogP (Wildman-Crippen) is chosen as the first target specifically because it is
an **atom-additive** property — RDKit provides exact per-atom contributions via
`Crippen._GetAtomContribs(mol)`. This gives a known ground truth to validate
the SHAP attributions against. If the per-atom SHAP values correlate with
Crippen atom contributions, the method is working correctly. If they diverge,
something is wrong.

After validating on CLogP, the method should be applied to a **non-additive**
property (e.g., mutagenicity/AMES) where no closed-form per-atom decomposition
exists. This is where the k-hop layer decomposition provides genuine new insight.

## Implementation Plan

### New module: `graph_hdc/shap.py`

Core utility functions for SHAP analysis on HDC embeddings.

```python
def get_player_components(node_hv_stack: np.ndarray) -> np.ndarray:
    """Reshape (num_nodes, depth+1, hidden_dim) -> (num_players, hidden_dim).

    Each row is one player h_i^(k). The sum of all rows equals the graph
    embedding (requires pooling='sum' in HyperNet).
    """

def make_coalition_predict_fn(components: np.ndarray, model) -> callable:
    """Return f(masks) -> predictions.

    masks: (n_samples, n_players) binary array
    Uses matrix multiply: embeddings = masks @ components
    Then calls model.predict(embeddings).
    """

def compute_shap_values(components: np.ndarray, model, nsamples='auto') -> np.ndarray:
    """Run KernelSHAP and return shap_values of shape (n_players,).

    Background: zero vector (empty coalition).
    Instance: all-ones vector (full coalition = full graph).
    nsamples defaults to 'auto' which uses 2 * n_players + 2048 internally.
    """

def compute_exact_shap_linear(components: np.ndarray, model) -> np.ndarray:
    """Compute exact Shapley values for a linear model (no sampling needed).

    For f(x) = w^T x + b, the Shapley value of player (i,k) is:
        φ_{i,k} = w^T h_i^(k) + b / n_players

    This serves as a verification oracle for the KernelSHAP pipeline.
    """

def aggregate_shap_by_node(shap_values, num_nodes, num_layers) -> np.ndarray:
    """Sum SHAP values across layers per node -> (num_nodes,).

    Preserves the efficiency property: sum of node values = f(full) - f(empty).
    """

def aggregate_shap_by_layer(shap_values, num_nodes, num_layers, absolute=True) -> np.ndarray:
    """Aggregate SHAP values across nodes per layer -> (num_layers,).

    If absolute=True: sum of |values| (importance magnitude, breaks efficiency).
    If absolute=False: sum of signed values (preserves efficiency, but
    cancellation across nodes may hide contributions).
    """

def get_player_labels(node_atoms, num_layers, atom_encoder=None) -> list[str]:
    """Return human-readable labels like 'C_0 (k=0)', 'O_3 (k=1)' for each player."""
```

Design notes:

- `make_coalition_predict_fn` is the key trick: since the readout is a sum,
  `mask @ components` directly produces the coalition embedding via matrix
  multiply.
- Everything operates on numpy arrays, matching `forward_graphs` output and
  sklearn model input.
- `compute_shap_values` must assert that `components` sum to the expected
  graph embedding (i.e., that `pooling='sum'` was used). This is the
  load-bearing invariant of the entire approach.
- `compute_exact_shap_linear` provides an analytical oracle for verifying
  that the KernelSHAP pipeline is wired up correctly (see Testing section).

### Modification: `HyperNet.forward` and `forward_graphs`

Add a `return_node_hv_stack: bool = False` flag to both `forward()` and
`forward_graphs()`. When `False` (default), `node_hv_stack` is not included in
the return dict — avoiding memory waste during normal encoding. When `True`,
it is included for SHAP analysis.

This avoids storing ~5 GB of intermediate embeddings for the full dataset.
During the SHAP experiment, only the few molecules selected for explanation are
re-encoded with this flag enabled.

```python
# In HyperNet.forward():
def forward(self, data, return_node_hv_stack=False):
    ...
    result = {
        'graph_embedding': embedding,
    }
    if return_node_hv_stack:
        result['node_hv_stack'] = node_hv_stack.transpose(0, 1)
    return result

# In HyperNet.forward_graphs():
def forward_graphs(self, graphs, batch_size=128, return_node_hv_stack=False):
    ...
    result = self.forward(data, return_node_hv_stack=return_node_hv_stack)
    ...
```

### New experiment: `graph_hdc/experiments/fingerprints/predict_molecules__hdc__shap__clogp.py`

Extends `predict_molecules__hdc__clogp.py`.

**Problem:** The `after_dataset` hook runs *before* model training (line 647 of
`predict_molecules.py`), so there is no existing post-training hook for SHAP
analysis.

**Solution:** Add a non-replacing `evaluate_model` hook (`replace=False`) that
runs after the base evaluation. It checks if the current model matches
`SHAP_MODEL_NAME` and only runs SHAP in that case. This avoids duplicating the
base metrics/plotting code.

#### Parameters

```python
DATASET_NAME: str = 'aqsoldb'
DATASET_TYPE: str = 'regression'
NUM_TEST: int = 100
EMBEDDING_SIZE: int = 2048
NUM_LAYERS: int = 2
SHAP_MODEL_NAME: str = 'random_forest'   # which trained model to explain
NUM_SHAP_MOLECULES: int = 10             # how many test molecules to explain
```

Note: `nsamples` defaults to `'auto'` in `compute_shap_values`, which uses
the SHAP library's internal heuristic (`2 * n_players + 2048`). This is
sufficient for stable estimates with ~90 players.

#### Hook overrides

**`process_dataset`** — Same as `predict_molecules__hdc.py`, unchanged. The
`node_hv_stack` is NOT stored for all molecules. Instead, the saved HyperNet
encoder path is stored on the experiment for later re-encoding.

**`evaluate_model` (replace=False)** — Appended after the base evaluation.
Checks `key == f'test_{e.SHAP_MODEL_NAME}'`. If so:

1. Reload the saved HyperNet encoder.
2. Select `NUM_SHAP_MOLECULES` test molecules.
3. Re-encode only those molecules with `return_node_hv_stack=True`.
4. For each molecule:
   - Extract components via `get_player_components(node_hv_stack)`.
   - Assert `components.sum(axis=0) ≈ graph_embedding` (pooling='sum' guard).
   - Run `compute_shap_values(components, model)`.
   - Aggregate by node and by layer.
   - Compare per-atom SHAP values with Crippen atom contributions.
5. Visualizations (see below).

#### Visualizations

- **2D molecular structure** with atoms colored by SHAP attribution magnitude,
  using `rdkit.Chem.Draw.MolToImage` with `highlightAtoms` and
  `highlightAtomColors`. This is the standard in molecular interpretability.
- **Per-atom bar chart** comparing SHAP attributions (signed) with Crippen atom
  contributions side by side.
- **Node x layer heatmap** showing the full `φ_{i,k}` decomposition — atoms on
  one axis, layers on the other.
- **Mean layer importance** bar chart aggregated across all explained molecules
  (using both signed and absolute aggregation).
- **Correlation scatter** of SHAP per-atom values vs Crippen per-atom values
  across all explained molecules.

### New tests: `tests/test_shap.py`

```python
class TestGetPlayerComponents:
    def test_shape(self):
        """(5 nodes, 3 layers, 100 dim) -> (15, 100)"""

    def test_sum_equals_embedding(self):
        """Sum of all components must equal graph_embedding from HyperNet.
        This validates the entire premise of the approach."""

    def test_single_node_graph(self):
        """Edge case: molecule with 1 atom produces (depth+1,) players."""

class TestCoalitionPredict:
    def test_full_coalition(self):
        """Full mask gives same prediction as full embedding."""

    def test_empty_coalition(self):
        """Empty mask predicts from zero vector."""

    def test_partial_mask(self):
        """Mask [1,0,1,0,...] produces sum of selected components only."""

class TestLinearModelOracle:
    def test_exact_shap_matches_kernel_shap(self):
        """For a LinearRegression model, compute_exact_shap_linear and
        compute_shap_values should produce matching results. This verifies
        the entire KernelSHAP pipeline is correctly wired."""

class TestShapValuesEfficiency:
    def test_shap_values_sum_property(self):
        """SHAP values sum to f(full) - f(empty), within tolerance."""

class TestAggregation:
    def test_aggregate_by_node(self):
        """Correct summation across layers per node."""

    def test_aggregate_by_layer_absolute(self):
        """Absolute summation across nodes per layer."""

    def test_aggregate_by_layer_signed(self):
        """Signed summation across nodes per layer, preserves efficiency."""

class TestPlayerLabels:
    def test_label_count(self):
        """Correct number of labels for given nodes and layers."""

    def test_label_format(self):
        """Labels match expected 'Symbol_i (k=K)' format."""
```

The most important tests:
- `test_sum_equals_embedding`: validates the additive decomposition premise.
- `test_exact_shap_matches_kernel_shap`: validates the full SHAP pipeline
  against an analytical oracle, independent of sampling noise.

### Dependency: `pyproject.toml`

Add `shap` as a required dependency (using Poetry format):

```toml
[tool.poetry.dependencies]
shap = ">=0.42"
```

## Files to create/modify

| File | Action | Purpose |
|------|--------|---------|
| `graph_hdc/shap.py` | Create | Core SHAP utilities |
| `graph_hdc/models.py` | Edit | Add `return_node_hv_stack` flag to `forward` and `forward_graphs` |
| `graph_hdc/experiments/fingerprints/predict_molecules__hdc__shap__clogp.py` | Create | CLogP + SHAP experiment |
| `tests/test_shap.py` | Create | Unit tests for core module |
| `pyproject.toml` | Edit | Add `shap` dependency |

## Considerations

**Pooling guard.** The additive decomposition only works when `pooling='sum'`.
The code must assert this at SHAP computation time. If `pooling='mean'`, the
coalition function `mask @ components` produces a sum, but the model was trained
on mean-pooled embeddings — the decomposition silently breaks.

**Out-of-distribution coalitions.** Partial sums are unusual inputs for the
trained model. Tree models (Random Forest) are relatively robust to this. As a
diagnostic, measure the cosine similarity between partial coalition embeddings
and the nearest training embedding to quantify how far out-of-distribution the
coalitions land. Training with random dropout of components could mitigate this
but is not needed for a first version.

**Normalization asymmetry.** Layer k=0 embeddings are un-normalized (from
property encoding), while layers k>=1 are L2-normalized (line 357 of
`models.py`). This means k=0 components may have different magnitude than k>=1
components. SHAP correctly accounts for this through the value function, but
users comparing raw SHAP magnitudes across layers should be aware of this
scale difference.

**Player overlap.** Neighboring nodes' k-hop embeddings share structural
information (e.g., `h_i^(1)` and `h_j^(1)` for neighbors i,j encode their
shared neighborhood). Shapley values handle this correctly — it is fair credit
allocation under redundancy — but the values should be interpreted as
context-dependent attributions, not independent importance scores.

**Aggregation semantics.** Summing signed SHAP values per node preserves the
efficiency property (they still sum to `f(full) - f(empty)`). Summing absolute
values per layer gives an importance magnitude measure that breaks efficiency.
Both are useful; the distinction should be documented and both options exposed.

**Future: mutagenicity.** After validating on CLogP (where ground truth exists),
apply to AMES mutagenicity where no atom-additive decomposition is known. This
is where the k-hop decomposition provides genuinely new interpretive value.
