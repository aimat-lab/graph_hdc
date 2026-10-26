"""
Training-free decoding of the atom composition of a molecule from its HDF vectors.

An HDF vector of a molecule superimposes three kinds of components: the bound hypervectors of its atoms
(element (x) heavy-atom degree (x) hydrogen count, the layer-0 node vectors), the bound neighborhood vectors of
the message-passing layers, and the encodings of the graph size and diameter. Since all codebook vectors are
known, parts of this information can be queried from the vector without any training:

- the number of atoms, together with the graph diameter, as the size and diameter whose bundled encodings are
  most similar to the vector (``decode_size_and_diameter``)
- the atom types and their numbers, by non-negative least squares on the bound atom-type hypervectors of the
  codebook (``atom_type_codebook``, ``decode_composition``)

The functions mirror ``HyperNet.encode_properties`` for the node encoders of
``make_molecule_node_encoder_map_cont`` ('node_atoms', 'node_degrees', 'node_valences').
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import nnls

from graph_hdc.binding import _circular_convolution_fft

#: An atom type as (atomic number, heavy-atom degree, hydrogen count).
AtomType = Tuple[int, int, int]


def atom_type_codebook(node_encoder_map: Dict[str, object],
                       degrees: Iterable[int] = range(5),
                       hydrogens: Iterable[int] = range(5),
                       ) -> Tuple[List[AtomType], np.ndarray]:
    """
    Returns all atom types of the codebook and their bound hypervectors.

    The types are all combinations of the elements of the 'node_atoms' encoder with the given heavy-atom degrees
    and hydrogen counts. Each hypervector is bound exactly like the layer-0 node vector of HyperNet, i.e. the
    circular convolution of the encodings of the three node properties in the order of ``node_encoder_map``.

    :param node_encoder_map: The node encoder map of the HyperNet ('node_atoms', 'node_degrees', 'node_valences').
    :param degrees: The heavy-atom degrees to include.
    :param hydrogens: The hydrogen counts to include.

    :returns: A tuple (types, matrix) with the list of atom types and the (num_types, dim) array of their
        hypervectors.
    """
    keys = list(node_encoder_map)
    if sorted(keys) != ['node_atoms', 'node_degrees', 'node_valences']:
        raise ValueError(f'expected the encoders node_atoms, node_degrees and node_valences, got {keys}')

    elements = [int(z) for z in node_encoder_map['node_atoms'].atoms]
    types = [(z, int(d), int(h)) for z in elements for d in degrees for h in hydrogens]
    values = {'node_atoms': {z: float(z) for z in elements},
              'node_degrees': {int(d): float(d) for d in degrees},
              'node_valences': {int(h): float(h) for h in hydrogens}}
    encodings = {key: {v: node_encoder_map[key].encode(torch.tensor(x, dtype=torch.float64)).to(torch.float64)
                       for v, x in values[key].items()}
                 for key in keys}

    position = {'node_atoms': 0, 'node_degrees': 1, 'node_valences': 2}
    rows = []
    for atom_type in types:
        stack = torch.stack([encodings[key][atom_type[position[key]]] for key in keys])
        rows.append(_circular_convolution_fft(stack))
    return types, torch.stack(rows).numpy()


def atom_composition(graph: dict, types: Sequence[AtomType]) -> np.ndarray:
    """
    Returns the number of atoms of every type in ``types`` for a graph dict of ``graph_dict_from_mol``.

    :raises KeyError: If an atom of the graph has a type that is not part of ``types``.
    """
    index = {atom_type: i for i, atom_type in enumerate(types)}
    counts = np.zeros(len(types))
    for z, d, h in zip(graph['node_atoms'], graph['node_degrees'], graph['node_valences']):
        counts[index[(int(z), int(d), int(h))]] += 1
    return counts


def decode_size_and_diameter(vector: np.ndarray,
                             size_encoder,
                             diameter_encoder,
                             max_size: int,
                             max_diameter: int,
                             ) -> Tuple[int, int]:
    """
    Returns the graph size (number of atoms) and graph diameter whose bundled encoding is most similar to
    ``vector``.

    HDF adds the normalized bundle of the size and diameter encodings to the normalized structural part, so the
    vector is compared with normalize(size(n) + diameter(m)) for every n <= ``max_size`` and m < n. The two
    attributes have to be decoded jointly: the fractional power encodings of small values of different
    attributes are similar to each other (the encoding of 0 is the identity of binding for every encoder), such
    that a query with the size encodings alone is biased by the diameter encoding.

    :param vector: The HDF vector of the molecule (it must contain the size and diameter encodings), or a
        (num_molecules, dim) array of such vectors.
    :param size_encoder: The 'graph_size' encoder of the graph encoder map.
    :param diameter_encoder: The 'graph_diameter' encoder of the graph encoder map.
    :param max_size: The largest number of atoms that is considered.
    :param max_diameter: The largest diameter that is considered.

    :returns: The tuple (size, diameter), or a tuple of two integer arrays for a 2D input.
    """
    vectors = np.atleast_2d(np.asarray(vector, dtype=np.float64))
    sizes, diameters = np.arange(1, max_size + 1), np.arange(0, max_diameter + 1)
    s = np.stack([size_encoder.encode(torch.tensor(float(n))).numpy() for n in sizes])
    d = np.stack([diameter_encoder.encode(torch.tensor(float(m))).numpy() for m in diameters])
    norms = np.sqrt((s ** 2).sum(1)[:, None] + (d ** 2).sum(1)[None, :] + 2 * s @ d.T)
    scores = ((vectors @ s.T)[:, :, None] + (vectors @ d.T)[:, None, :]) / norms[None]
    scores[:, diameters[None, :] >= sizes[:, None]] = -np.inf   # the diameter is smaller than the number of atoms
    flat = scores.reshape(len(vectors), -1).argmax(axis=1)
    i, j = np.unravel_index(flat, norms.shape)
    if np.ndim(vector) == 1:
        return int(sizes[i[0]]), int(diameters[j[0]])
    return sizes[i], diameters[j]


def integer_counts(coefficients: np.ndarray, total: int, iterations: int = 100) -> np.ndarray:
    """
    Converts non-negative coefficients that are proportional to atom counts into integer counts.

    The coefficients are divided by the smallest scale for which the rounded values sum to at most ``total``
    (bisection; the rounded sum decreases monotonically with the scale), so that coefficients far below the
    scale of a single atom round to zero.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if total <= 0 or coefficients.max(initial=0.0) <= 0:
        return np.zeros_like(coefficients)
    low, high = 0.0, 2.0 * coefficients.max()
    for _ in range(iterations):
        middle = (low + high) / 2
        if np.round(coefficients / middle).sum() > total:
            low = middle
        else:
            high = middle
    return np.round(coefficients / high)


def decode_composition(vector: np.ndarray,
                       codebook: np.ndarray,
                       num_atoms: Optional[int] = None,
                       shortlist: int = 64,
                       ) -> np.ndarray:
    """
    Decodes the number of atoms of every codebook type from an HDF vector without any training.

    The ``shortlist`` atom types whose hypervectors are most similar to the vector are kept, and their
    coefficients are fitted by non-negative least squares, which resolves the overlap between similar atom
    types (neighboring degrees and hydrogen counts have similar fractional power encodings). The coefficients
    are then converted to integer counts: with ``num_atoms``, by the scale for which the counts sum to
    ``num_atoms`` (for normalized vectors, whose scale is unknown); without it, by rounding directly (for the
    unnormalized layer-0 readout, in which every atom contributes one unit-norm vector).

    :param vector: The HDF vector to decode.
    :param codebook: The (num_types, dim) array of atom-type hypervectors of ``atom_type_codebook``.
    :param num_atoms: The number of atoms of the molecule (e.g. from ``decode_size_and_diameter``), or None.
    :param shortlist: The number of candidate types for the least-squares fit.

    :returns: The decoded count of every codebook type.
    """
    vector = np.asarray(vector, dtype=np.float64)
    candidates = np.argsort(-(codebook @ vector))[:shortlist]
    coefficients = np.zeros(len(codebook))
    coefficients[candidates], _ = nnls(codebook[candidates].T, vector, maxiter=50 * len(candidates))
    if num_atoms is None:
        return np.round(coefficients)
    return integer_counts(coefficients, num_atoms)


def composition_metrics(decoded: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Compares a decoded composition with the true one.

    :returns: A dict with 'exact_counts' (all counts correct), 'exact_set' (the same atom types present,
        regardless of their numbers) and 'atoms_correct' (the share of atoms that are assigned the correct type,
        i.e. the sum of min(decoded, true) over the types divided by the number of atoms).
    """
    decoded, true = np.asarray(decoded), np.asarray(true)
    return {
        'exact_counts': float(np.array_equal(decoded, true)),
        'exact_set': float(np.array_equal(decoded > 0, true > 0)),
        'atoms_correct': float(np.minimum(decoded, true).sum() / true.sum()),
    }
