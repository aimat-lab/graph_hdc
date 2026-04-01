# CompositeHyperNet Implementation Summary

## Overview

`CompositeHyperNet` is a variant of `HyperNet` that creates composite embeddings with explicit separation of structural information at different levels. This implementation was completed on 2025-10-14.

## What was Implemented

### 1. Core CompositeHyperNet Class (`graph_hdc/models.py`)

The `CompositeHyperNet` class inherits from `HyperNet` and modifies the embedding scheme to create a composite vector with three components:

- **h_0** (Order-0): Sum of initial node representations
  - Captures "bag of nodes" information
  - Makes node decoding straightforward and accurate

- **h_1** (Order-1): Sum of initial edge representations
  - Captures "bag of edges" information
  - Makes edge decoding independent and more accurate

- **g** (Global): Standard HyperNet embedding after message passing
  - Preserves global context and higher-order structural patterns
  - Used for distance calculations during reconstruction

**Final embedding**: `h_0 | h_1 | g` (concatenation)
- Shape: `(batch_size, 3 * hidden_dim)`

### 2. Key Methods Implemented

#### `__init__(**kwargs)`
Delegates all initialization to parent `HyperNet` class for full compatibility.

#### `forward(data: Data) -> dict`
Creates composite embeddings with explicit component separation:
1. Encodes node/graph properties using parent's `encode_properties()`
2. Computes h_0 as sum of initial node hypervectors
3. Computes h_1 as sum of bound edge hypervectors
4. Performs message passing to compute g
5. Concatenates h_0 | h_1 | g

Returns:
- `graph_embedding`: Composite vector (batch, 3*hidden_dim)
- `graph_hv_stack`: Per-layer composite embeddings
- `h_0`, `h_1`, `g`: Individual components

#### `decode_order_zero(embedding: torch.Tensor) -> List[dict]`
Decodes nodes directly from h_0 component:
- More accurate than standard HyperNet (pure node information)
- Simpler algorithm (no layer handling needed)
- Faster execution

#### `decode_order_one(embedding: torch.Tensor, ...) -> List[dict]`
Decodes edges directly from h_1 component:
- More accurate than standard HyperNet (pure edge information)
- No correction factors needed (clean information)
- Can work without node constraints (computes internally if needed)

#### `extract_distance_embedding(embedding: torch.Tensor) -> torch.Tensor`
**NEW METHOD** - Critical for reconstruction:
- Extracts the `g` component for distance calculations
- Only g should be used during reconstruction (not h_0 or h_1)
- Also added to base `HyperNet` class (returns full embedding)

### 3. Usage for Reconstruction

When using `CompositeHyperNet` with reconstruction algorithms, you should:

```python
from graph_hdc.models import CompositeHyperNet
from graph_hdc.reconstruct import GraphReconstructor

# 1. Create encoder
encoder = CompositeHyperNet(
    hidden_dim=5000,
    depth=3,
    node_encoder_map=...
)

# 2. Encode target graph
target_result = encoder.forward_graphs([target_graph])[0]
target_embedding = torch.tensor(target_result['graph_hv_stack'])

# 3. For distance calculations, extract g component
# This is automatically handled if you use encoder.extract_distance_embedding()
# But reconstruction algorithms should be updated to use this method

# 4. Use with reconstructor
reconstructor = GraphReconstructor(encoder=encoder, ...)
result = reconstructor.reconstruct(embedding=target_embedding)
```

**Important Note for Reconstruction Algorithm Developers:**

When computing distances between embeddings, use:
```python
# CORRECT - uses appropriate component (g for CompositeHyperNet)
target_dist_emb = encoder.extract_distance_embedding(target_embedding)
candidate_dist_emb = encoder.extract_distance_embedding(candidate_embedding)
distance = distance_func(target_dist_emb, candidate_dist_emb)

# INCORRECT - compares full composite embedding (includes h_0, h_1)
distance = distance_func(target_embedding, candidate_embedding)  # Don't do this!
```

## Test Coverage

### test_composite_hypernet.py (18 tests)
- **Basics** (4 tests): Initialization, forward pass, shapes, batching
- **Decoding** (4 tests): Order-0 decoding, order-1 decoding, independence
- **Edge Cases** (4 tests): Single nodes, no edges, bidirectional, 2D embeddings
- **Distance Extraction** (4 tests): 1D/2D extraction, HyperNet compatibility, reconstruction usage
- **Comparison** (2 tests): Comparison with standard HyperNet, accuracy

### test_composite_hypernet_integration.py (2 tests)
- Integration with forward_graphs
- Decode methods with reconstruction format

### test_composite_hypernet_molecules.py (4 tests)
- Real molecular encoding
- Molecular decoding
- Batch processing
- Component verification

**Total: 24 tests, all passing**

## Key Design Decisions

### 1. Inheritance from HyperNet
Maximizes code reuse and ensures compatibility with existing infrastructure.

### 2. Composite Embedding Structure
h_0 | h_1 | g concatenation provides:
- Explicit separation for better decoding
- Preservation of global context (g)
- 3x larger embeddings (acceptable trade-off)

### 3. Distance Calculations on g Only
Critical insight: During reconstruction, only the g component should be compared because:
- h_0 captures only node counts (not structural similarity)
- h_1 captures only edge types (not connectivity patterns)
- g captures global structure after message passing

### 4. No Correction Factors for CompositeHyperNet
Unlike standard HyperNet, decode_order_one doesn't need correction factors because h_1 contains pure edge information without interference from message passing.

## Benefits Over Standard HyperNet

1. **More Accurate Decoding**: Direct access to h_0 and h_1 improves reconstruction
2. **Independent Edge Decoding**: h_1 allows edge decoding without needing node constraints
3. **Cleaner Information**: Each component has specific semantic meaning
4. **Better for Analysis**: Can inspect node/edge/global contributions separately
5. **Improved Reconstruction**: Explicit structure aids in guided search algorithms

## Backward Compatibility

- Fully compatible with existing HyperNet infrastructure
- Same initialization parameters
- Returns additional keys in forward() output (backward compatible)
- decode_order_zero/one have same signatures
- extract_distance_embedding added to base class (no breaking changes)

## Future Work

Potential enhancements:
1. Update GraphReconstructor to use extract_distance_embedding()
2. Update GraphReconstructorAStar to use extract_distance_embedding()
3. Experiment with weighted combinations of h_0, h_1, g for distance
4. Add option to include edge weights in h_1 computation
5. Explore using h_0 and h_1 for specialized reconstruction strategies

## References

- Base implementation: `graph_hdc/models.py` (lines 944-1449)
- Tests: `tests/test_composite_hypernet.py`
- Integration tests: `tests/test_composite_hypernet_integration.py`
- Molecular tests: `tests/test_composite_hypernet_molecules.py`

## Authors

Implemented with Claude Code (Anthropic) on 2025-10-14
