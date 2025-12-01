# Norm Regularization Comparison: Explicit vs Implicit

## Overview

When optimizing HDC embeddings (which are normalized to unit norm), there are two approaches to maintain the norm constraint during gradient descent:

### 1. Explicit Projection (Hard Constraint)
- **Parameter**: `USE_NORM_PROJECTION: true`
- **Mechanism**: After each `optimizer.step()`, explicitly project the representation back to unit norm
- **Guarantees**: Representation norm = 1.0 at all times
- **Flow weight**: Can be small (0.1) since projection handles the constraint

### 2. Implicit Regularization (Soft Constraint)
- **Parameter**: `USE_NORM_PROJECTION: false`
- **Mechanism**: Flow model's likelihood provides gradient that encourages norm ≈ 1.0
- **Guarantees**: None - relies on optimization dynamics
- **Flow weight**: Must be large (≥ 5.0) to provide sufficient regularization

## Why This Matters

HDC embeddings from `HyperNet` with `normalize_all=True` are inherently unit-norm vectors:
- Training data for flow model has norm ≈ 1.0
- Flow model learns distribution p(x) where ||x|| ≈ 1.0
- If representation drifts to ||x|| >> 1.0 during optimization, flow model returns very negative (or NaN) log probabilities

## Trade-offs

### Explicit Projection
**Pros:**
- Guaranteed constraint satisfaction
- Numerically stable even with small flow weights
- Simpler to tune (fewer hyperparameters)
- Safe default for HDC embeddings

**Cons:**
- Projection may conflict with gradient direction
- Less "pure" optimization (not just gradient descent)
- Doesn't test whether flow model alone can regularize

### Implicit Regularization
**Pros:**
- Pure gradient-based optimization (no hard constraints)
- Tests the flow model's ability to regularize
- Theoretically elegant
- May find better local optima if flow weight is well-tuned

**Cons:**
- Requires careful tuning of `FLOW_WEIGHT`
- Risk of numerical instability if norm drifts too far
- May produce NaN if flow weight is insufficient
- More hyperparameters to tune

## Experimental Comparison

### Configuration 1: Explicit (Default)
**File**: `optimize_molecule_nf__hdc__clogp.yml`

```yaml
FLOW_WEIGHT: 0.1
USE_NORM_PROJECTION: true
```

**Expected behavior:**
- `representation_norm` stays exactly at 1.0
- No NaN values
- Flow loss contributes but doesn't dominate
- Optimization primarily driven by MSE loss

### Configuration 2: Implicit (Experimental)
**File**: `optimize_molecule_nf__hdc__clogp_implicit.yml`

```yaml
FLOW_WEIGHT: 5.0
USE_NORM_PROJECTION: false
```

**Expected behavior:**
- `representation_norm` fluctuates around 1.0 (e.g., 0.95 - 1.05)
- Flow loss dominates when norm drifts from 1.0
- Should not produce NaN if flow weight is sufficient
- May converge to different local optima

## Recommendations

1. **Default approach**: Use explicit projection (`USE_NORM_PROJECTION: true`)
   - Safer and more robust
   - Works well with small flow weights
   - Good for production experiments

2. **Experimental approach**: Try implicit regularization for comparison
   - Increase `FLOW_WEIGHT` to 5.0 or higher
   - Monitor `representation_norm` in optimization history
   - If norms drift significantly (> 1.5), increase flow weight further

3. **Hybrid approach**: Use moderate flow weight + projection
   - `FLOW_WEIGHT: 1.0` and `USE_NORM_PROJECTION: true`
   - Flow loss helps guide optimization toward in-distribution regions
   - Projection ensures numerical stability

## Implementation Details

The key code in `optimize_molecule_nf.py` (lines 1095-1100):

```python
optimizer.step()

# Optionally project representation back to unit norm
if e.USE_NORM_PROJECTION:
    with torch.no_grad():
        representation.data = representation.data / torch.norm(representation.data)

scheduler.step()
```

The optimization history tracks `representation_norm` to monitor behavior (line 1113).

## Testing Both Approaches

```bash
# Test explicit projection (safe default)
python -m pycomex run optimize_molecule_nf__hdc__clogp.yml

# Test implicit regularization (experimental)
python -m pycomex run optimize_molecule_nf__hdc__clogp_implicit.yml
```

Compare the results:
- Check for NaN values
- Compare final property predictions
- Examine `representation_norm` trajectories
- Compare reconstruction quality (Tanimoto similarity)
- Review `flow_training_history.png` to verify flow training converged properly
