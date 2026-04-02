"""
Transition Metal Complex (TMC) utilities for hyperdimensional computing.

This module provides functions to decompose pre-processed TMC graph dicts (as produced
by ``chem_mat_data.tmc_processing.MetalOrganicProcessing``) into separate symbolic
property arrays suitable for HDC encoding via :class:`~graph_hdc.models.HyperNet`.

It mirrors :mod:`graph_hdc.special.molecules` but targets the richer feature set of
transition metal complexes: metal-center flags, d-electron counts, formal charges,
and a broader element palette (50 elements covering organic atoms and 3d/4d/5d
transition metals).

Usage example:

.. code-block:: python

    from chem_mat_data.tmc import load_tmc_dataset
    from chem_mat_data.tmc_processing import MetalOrganicProcessing
    from graph_hdc.special.tmcs import (
        graph_dict_from_tmc_graph,
        make_tmc_node_encoder_map_cont,
        make_tmc_graph_encoder_map_cont,
    )
    from graph_hdc.models import HyperNet

    processing = MetalOrganicProcessing()
    df = load_tmc_dataset('tmqmg')
    row = df.iloc[0]
    graph = processing.process(
        metal=row['metal'],
        ligand_smiles=row['ligand_smiles'],
        connecting_atom_indices=row['connecting_atom_indices'],
        total_charge=row['total_charge'],
    )
    graph = graph_dict_from_tmc_graph(graph)

    encoder = HyperNet(
        hidden_dim=2048,
        depth=2,
        node_encoder_map=make_tmc_node_encoder_map_cont(dim=2048),
        graph_encoder_map=make_tmc_graph_encoder_map_cont(dim=2048),
    )
    results = encoder.forward_graphs([graph])
"""
from typing import List, Dict

import copy
import numpy as np
import networkx as nx
from rdkit.Chem import GetPeriodicTable

from chem_mat_data.tmc_processing import TMC_ELEMENTS

from graph_hdc.special.molecules import AtomEncoder
from graph_hdc.utils import AbstractEncoder
from graph_hdc.utils import CategoricalIntegerEncoder
from graph_hdc.utils import ContinuousEncoder

_pt = GetPeriodicTable()

#: Atomic numbers corresponding to :data:`TMC_ELEMENTS`, in the same order.
#: Used to initialise :class:`AtomEncoder` so that the encoder keys match the
#: numeric ``node_atoms`` values produced by :func:`graph_dict_from_tmc_graph`.
TMC_ATOMIC_NUMBERS: List[int] = [_pt.GetAtomicNumber(sym) for sym in TMC_ELEMENTS]

# Offset applied to raw formal charges so that negative values (e.g. -3) map
# to non-negative category indices for ``CategoricalIntegerEncoder``.
FORMAL_CHARGE_OFFSET: int = 3
# Number of categories for formal charge: covers range [-3, +6] -> [0, 9]
FORMAL_CHARGE_NUM_CATEGORIES: int = 10


def graph_dict_from_tmc_graph(
    graph: dict,
    tmc_elements: List[str] = TMC_ELEMENTS,
) -> dict:
    """
    Decompose a pre-processed TMC graph dict into separate symbolic property arrays.

    The input ``graph`` is expected to come from
    ``chem_mat_data.tmc_processing.MetalOrganicProcessing.process()`` (or from an
    mpack loaded via ``load_graph_dataset``).  Its ``node_attributes`` is a dense
    ``(num_atoms, 91)`` array.  This function extracts individual properties into
    top-level keys so that :class:`~graph_hdc.models.HyperNet` can read them via
    its encoder map.

    The following keys are added/overwritten:

    * ``node_atoms`` -- atomic numbers (float array), decoded from the symbol
      one-hot at positions ``[0:51]``.
    * ``node_degrees`` -- integer degrees (float array), decoded from the
      one-hot at positions ``[61:74]``.
    * ``node_is_metal`` -- binary flag (float array) from position ``[83]``.
    * ``node_formal_charge`` -- formal charge **offset by +3** and clamped to
      ``[0, 9]`` (float array) so it can be used with a 10-category
      ``CategoricalIntegerEncoder``.
    * ``node_d_electrons`` -- d-electron count in range 0-10 (float array),
      denormalised from position ``[89]``.
    * ``graph_size`` -- number of nodes (float scalar).
    * ``graph_diameter`` -- graph diameter (float scalar), computed on the
      undirected graph.  If the graph is disconnected the diameter of the
      largest connected component is used.
    * ``graph_total_charge`` -- total complex charge (float scalar) from
      ``graph_attributes[1]``.

    :param graph: A TMC graph dict with at least ``node_attributes``,
        ``node_indices``, ``edge_indices``, and ``graph_attributes``.
    :param tmc_elements: Element symbol list matching the one-hot encoding
        order in ``node_attributes[:, 0:51]``.

    :returns: A **copy** of the input dict enriched with the decomposed
        property arrays.
    """
    result = copy.deepcopy(graph)
    node_attrs = graph['node_attributes']  # (num_atoms, 91)
    num_nodes = node_attrs.shape[0]

    # --- Atomic numbers from symbol one-hot [0:51] ---
    symbol_onehot = node_attrs[:, 0:51]
    node_atoms = np.zeros(num_nodes, dtype=float)
    for i in range(num_nodes):
        idx = int(np.argmax(symbol_onehot[i]))
        if idx < len(tmc_elements):
            node_atoms[i] = float(_pt.GetAtomicNumber(tmc_elements[idx]))
        else:
            # Unknown element -- use 0
            node_atoms[i] = 0.0
    result['node_atoms'] = node_atoms

    # --- Degrees from one-hot [61:74] (13 categories: 0-12) ---
    degree_onehot = node_attrs[:, 61:74]
    node_degrees = np.array(
        [float(np.argmax(degree_onehot[i])) for i in range(num_nodes)],
        dtype=float,
    )
    result['node_degrees'] = node_degrees

    # --- is_metal_center from [83] ---
    result['node_is_metal'] = node_attrs[:, 83].astype(float)

    # --- formal_charge from [80], offset by +3 and clamped to [0, 9] ---
    raw_charge = node_attrs[:, 80]
    result['node_formal_charge'] = np.clip(
        raw_charge + FORMAL_CHARGE_OFFSET, 0, FORMAL_CHARGE_NUM_CATEGORIES - 1,
    ).astype(float)

    # --- d_electron_count from [89], denormalized (*10) ---
    result['node_d_electrons'] = np.round(node_attrs[:, 89] * 10.0).astype(float)

    # --- Graph-level properties ---
    edge_indices = graph['edge_indices']
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i, j in edge_indices:
        G.add_edge(int(i), int(j))

    result['graph_size'] = float(num_nodes)

    if num_nodes <= 1 or G.number_of_edges() == 0:
        result['graph_diameter'] = 0.0
    elif nx.is_connected(G):
        result['graph_diameter'] = float(nx.diameter(G))
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        result['graph_diameter'] = float(nx.diameter(G.subgraph(largest_cc)))

    result['graph_total_charge'] = float(graph['graph_attributes'][1])

    return result


def make_tmc_node_encoder_map(
    dim: int,
    atoms: List[int] = TMC_ATOMIC_NUMBERS,
    seed: int = 0,
) -> Dict[str, AbstractEncoder]:
    """
    Create a categorical-only node encoder map for TMC graphs.

    All properties are encoded with ``CategoricalIntegerEncoder`` (or
    ``AtomEncoder`` for atoms).  This is the mode required for decoding /
    reconstruction.

    :param dim: Hypervector dimensionality.
    :param atoms: List of atomic numbers to support. Defaults to
        :data:`TMC_ATOMIC_NUMBERS` (50 elements).
    :param seed: Base random seed for reproducibility.

    :returns: Encoder map compatible with :class:`~graph_hdc.models.HyperNet`.
    """
    return {
        'node_atoms': AtomEncoder(dim=dim, atoms=atoms, seed=seed),
        'node_degrees': CategoricalIntegerEncoder(dim=dim, num_categories=13, seed=seed + 10),
        'node_is_metal': CategoricalIntegerEncoder(dim=dim, num_categories=2, seed=seed + 20),
        'node_formal_charge': CategoricalIntegerEncoder(
            dim=dim, num_categories=FORMAL_CHARGE_NUM_CATEGORIES, seed=seed + 30,
        ),
        'node_d_electrons': CategoricalIntegerEncoder(dim=dim, num_categories=11, seed=seed + 40),
    }


def make_tmc_node_encoder_map_cont(
    dim: int,
    atoms: List[int] = TMC_ATOMIC_NUMBERS,
    seed: int = 0,
) -> Dict[str, AbstractEncoder]:
    """
    Create a node encoder map for TMC graphs with continuous degree encoding.

    Uses ``ContinuousEncoder`` for ``node_degrees`` (FHRR encoding preserves
    ordinal relationships) and ``CategoricalIntegerEncoder`` for discrete
    properties (atoms, is_metal, formal_charge, d_electrons).

    :param dim: Hypervector dimensionality.
    :param atoms: List of atomic numbers to support.
    :param seed: Base random seed for reproducibility.

    :returns: Encoder map compatible with :class:`~graph_hdc.models.HyperNet`.
    """
    return {
        'node_atoms': AtomEncoder(dim=dim, atoms=atoms, seed=seed),
        'node_degrees': ContinuousEncoder(dim=dim, size=13.0, bandwidth=3.0, seed=seed + 10),
        'node_is_metal': CategoricalIntegerEncoder(dim=dim, num_categories=2, seed=seed + 20),
        'node_formal_charge': CategoricalIntegerEncoder(
            dim=dim, num_categories=FORMAL_CHARGE_NUM_CATEGORIES, seed=seed + 30,
        ),
        'node_d_electrons': CategoricalIntegerEncoder(dim=dim, num_categories=11, seed=seed + 40),
    }


def make_tmc_graph_encoder_map_cont(
    dim: int,
    max_graph_size: float = 80.0,
    max_graph_diameter: float = 30.0,
    max_total_charge: float = 6.0,
    seed: int = 0,
) -> Dict[str, AbstractEncoder]:
    """
    Create a graph-level encoder map for TMC graphs.

    Uses ``ContinuousEncoder`` (FHRR) for all three properties: graph size,
    graph diameter, and total complex charge.

    :param dim: Hypervector dimensionality.
    :param max_graph_size: Upper bound for the graph-size encoder range.
    :param max_graph_diameter: Upper bound for the diameter encoder range.
    :param max_total_charge: Upper bound (absolute value) for the charge
        encoder range.
    :param seed: Base random seed for reproducibility.

    :returns: Encoder map compatible with :class:`~graph_hdc.models.HyperNet`.
    """
    return {
        'graph_size': ContinuousEncoder(
            dim=dim,
            size=max_graph_size,
            bandwidth=max(3.0, max_graph_size / 7.0),
            seed=seed,
        ),
        'graph_diameter': ContinuousEncoder(
            dim=dim,
            size=max_graph_diameter,
            bandwidth=max(2.0, max_graph_diameter / 5.0),
            seed=seed + 10,
        ),
        'graph_total_charge': ContinuousEncoder(
            dim=dim,
            size=max_total_charge,
            bandwidth=max(1.0, max_total_charge / 4.0),
            seed=seed + 20,
        ),
    }
