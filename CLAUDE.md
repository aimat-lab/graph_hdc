# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **hyperdimensional computing (HDC)** for graph structures. The project uses message passing networks to encode graphs into high-dimensional hypervectors that can be decoded to recover structural information. It focuses on molecular property prediction and graph generation tasks.

## Paper submission artifact — Code Ocean capsule (`../hdf-experiments`)

The paper **"Hyper-Dimensional Fingerprints as Molecular Representations"** (preprint
arXiv:2604.27810; target journal **Nature Computational Science**; method package on
Zenodo `10.5281/zenodo.19373621`) is accompanied by a **separate, standalone git repo**
that is the reproducibility artifact:

- **Location:** `/media/ssd2/Programming/hdf-experiments` (its own git repo, branch `main`; not a subdir of this repo).
- **What it is:** a [Code Ocean](https://codeocean.com) *compute capsule* — a curated,
  self-contained copy of this project holding **only** the experiment families the paper
  reports (`predict_molecules`, `molecule_similarity`, `optimize_molecule_bo`,
  `predict_bioactivity`). The `graph_hdc` library is **vendored** into it (not installed).
- **Layout (Code Ocean convention):**
  - `code/graph_hdc/` — vendored library · `code/experiments/fingerprints/` — the in-scope
    experiment scripts + YAML configs + `mixin_clogp.py` · `code/tests/`
  - `code/run` — the single headless entry point ("Reproducible Run"); tiers
    `TIER=smoke|demo|full` (demo is the default scaled-down run that fits Code Ocean;
    full needs a cluster).
  - `code/collect_outputs.py`, `code/experiments/fingerprints/figure_style.py` +
    `make_figure_{ged,bo,prediction}.py` — produce a metrics `summary.md` and the **three
    curated paper figures** (the experiments' own exploratory plots are not surfaced).
  - `environment/Dockerfile` (+ `postInstall`) — pinned CPU stack (python 3.11, torch,
    rdkit, pycomex, `chem_mat_database` + `vgd_counterfactuals` git deps). Local image tag: `hdf-artifact`.
  - `data/` (datasets auto-fetched, nothing committed) · `metadata/metadata.yml`.
- **Caveat:** the capsule vendors its own copy of `graph_hdc` and the experiment scripts, so
  edits made *here* do **not** propagate there. If a paper-relevant experiment or the
  library changes, re-sync the affected files into `../hdf-experiments`.

## Virtual Environment

This project uses a virtual environment which should be activated before running any command line tools or scripts.
To activate the virtual environment, run:

```bash
source .venv/bin/activate
```

## Development Guidelines

### Docstrings

Docstrings should use the ReStructuredText (reST) format. This is important for generating documentation and for consistency across the codebase. Docstrings should always start with a one-line summary followed by a more detailed paragraph - also including usage examples, for instance. If appropriate, docstrings should not only describe a method or function but also shed some light on the design rationale.

Documentation should also be *appropriate* in length. For simple functions, a brief docstring is sufficient. For more complex functions or classes, more detailed explanations and examples should be provided.

An example docstring may look like this:

```python

def multiply(a: int, b: int) -> int:
    """
    Multiply two integers `a` and `b`.

    This function takes two integers as input and returns their product.

    Example:
    
    ... code-block:: python

        result = multiply(3, 4)
        print(result)  # Output: 12

    :param a: The first integer to multiply.
    :param b: The second integer to multiply.

    :return: The product of the two integers.
    """
    return a * b

```

## Core Architecture

### Main Components

- **HyperNet models** (`graph_hdc/models.py`): Core hyperdimensional neural networks including `AbstractHyperNet` base class and specific implementations
- **Graph utilities** (`graph_hdc/graph.py`): Graph representation handling, constraint evaluation, and PyTorch Geometric integration
- **Binding operations** (`graph_hdc/binding.py`): Core HDC binding functions like circular convolution for combining hypervectors
- **Encoders** (`graph_hdc/utils.py`): Property encoders (categorical, numerical) for converting graph attributes to hypervectors
- **Special modules** (`graph_hdc/special/`): Domain-specific utilities for molecules and color graphs
- **Unittests** (`tests/`): Comprehensive tests for all components using `pytest`
- **Experiment scripts** (`graph_hdc/experiments/fingerprints/`): Experimentation framework using PyComex for molecular property prediction tasks

### Key Concepts

- **Hypervectors**: High-dimensional vectors (typically 1000+ dimensions) representing graph elements
- **Message Passing**: Graph neural network paradigm adapted for HDC
- **Binding**: Mathematical operations to combine hypervectors while preserving information
- **Encoding/Decoding**: Converting between symbolic graph properties and hypervector representations

## Commands

### Development Environment

This project uses a virtual environment for dependency management. The virtual environment should be activated before every command.

```bash
source .venv/bin/activate
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