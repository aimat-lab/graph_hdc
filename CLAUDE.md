# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **hyperdimensional computing (HDC)** for graph structures. The project uses message passing networks to encode graphs into high-dimensional hypervectors that can be decoded to recover structural information. It focuses on molecular property prediction and graph generation tasks.

## Core Architecture

### Main Components

- **HyperNet models** (`graph_hdc/models.py`): Core hyperdimensional neural networks including `AbstractHyperNet` base class and specific implementations
- **Graph utilities** (`graph_hdc/graph.py`): Graph representation handling, constraint evaluation, and PyTorch Geometric integration
- **Binding operations** (`graph_hdc/binding.py`): Core HDC binding functions like circular convolution for combining hypervectors
- **Encoders** (`graph_hdc/utils.py`): Property encoders (categorical, numerical) for converting graph attributes to hypervectors
- **Special modules** (`graph_hdc/special/`): Domain-specific utilities for molecules and color graphs

### Key Concepts

- **Hypervectors**: High-dimensional vectors (typically 1000+ dimensions) representing graph elements
- **Message Passing**: Graph neural network paradigm adapted for HDC
- **Binding**: Mathematical operations to combine hypervectors while preserving information
- **Encoding/Decoding**: Converting between symbolic graph properties and hypervector representations

## Commands

### Development Environment

```bash
# Setup virtual environment and UV (from DEVELOP.rst)
uv venv --seed --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
```

### Testing

```bash
# Run tests excluding local-only tests
pytest -q -m "not localonly"
```

## Experiment Structure

The project uses **PyComex** framework for computational experiments. Experiments are in `graph_hdc/experiments/fingerprints/` and follow naming convention:
- `predict_molecules__hdc__[dataset].py` - HDC-based experiments
- `predict_molecules__gnn__[dataset].py` - Graph neural network baselines  
- `predict_molecules__fp__[dataset].py` - Traditional fingerprint baselines

Common datasets include: ames, bace, bbbp, clogp, aqsoldb, qm9_smiles, zinc250

## Code Conventions

1. **Documentation**: Use ReStructuredText docstrings with parameter descriptions
2. **Type Hints**: Required for all functions/methods
3. **Function Parameters**: Split long parameter lists across multiple lines
4. **Testing**: Use pytest with markers (e.g., `@pytest.mark.localonly` for tests requiring special setup)

## Architecture Patterns

### Hypervector Operations
- Use `AbstractEncoder` subclasses for property encoding
- Binding operations combine hypervectors while preserving structure
- Decoding operations extract constraints/properties from hypervectors

### Graph Processing
- Convert NetworkX graphs to internal dict format using `graph_hdc.graph` utilities
- Use PyTorch Geometric `Data` objects for neural network processing
- Batch processing through `DataLoader` for efficiency

### Experiment Design
- Inherit from PyComex `Experiment` class
- Use experiment hooks for reusable components
- Store results in experiment-specific artifact directories
- Support both debug and production modes via `__DEBUG__` flag

## Dependencies

Key dependencies include:
- PyTorch & PyTorch Lightning (neural networks)
- PyTorch Geometric (graph operations)
- PyComex (experiment framework)
- RDKit (molecular chemistry)
- scikit-learn (ML baselines)
- Rich Click (CLI interface)