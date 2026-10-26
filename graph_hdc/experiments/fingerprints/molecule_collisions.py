"""
Collisions of molecular representations: HDF, binary Morgan fingerprints and the 1-WL test (experiment ex_16).

For every molecule of a dataset, we check whether its representation is shared with at least one other, distinct
molecule (distinct = different canonical SMILES without stereochemistry). The collision rate of a representation is
the fraction of molecules for which this is the case. The experiment supports the response to reviewer comment R3.2
on the expressivity of HDF relative to the 1-WL test.

* 1-WL (size-free reference): Weisfeiler-Leman graph hashes after WL_ITERATIONS iterations, (a) on the graph that HDF
  encodes (atoms labeled by atomic number, implicit hydrogen count and heavy-atom degree, bonds unlabeled; elements
  outside the HDF element list share one label, as in HDF), and (b) on the same graph with bond types as edge labels.
* Morgan: binary Morgan fingerprints with radius MORGAN_RADIUS, folded to each embedding size D. A collision is a
  pair of identical bit vectors.
* HDF: bidirectional HDF with NUM_LAYERS message-passing steps at each D. A collision is a pair of vectors that are
  identical up to floating-point precision, never merely similar molecules. The numerical noise floor is measured by
  encoding molecules with randomly permuted atom orders, and the collision tolerance is TOLERANCE_FACTOR times that
  floor. All pairs within the larger SEARCH_RADIUS are found exactly by sorting random one-dimensional projections
  (a pair at Euclidean distance d differs by at most d in every projection onto a unit vector), so that pairs which
  are close but not numerically identical are reported separately instead of being counted as collisions.

Each run handles one dataset and the embedding sizes in EMBEDDING_SIZES; the 1-WL hashes are computed in every run
and cross-tabulated with the HDF collisions.
"""
import inspect
import random
import time
from collections import Counter, defaultdict
from typing import List, Optional

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data import load_smiles_dataset

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (graph_dict_from_mol, make_molecule_node_encoder_map_cont,
                                         make_molecule_graph_encoder_map_cont)

RDLogger.DisableLog('rdApp.*')

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       Name of the chem_mat_data SMILES dataset (e.g. "qm9_smiles", "zinc250k").
DATASET_NAME: str = 'qm9_smiles'
# :param NUM_DATA:
#       If given, a random subset of this many distinct molecules is used (for smoke tests). None uses all molecules.
NUM_DATA: Optional[int] = None

# == REPRESENTATION PARAMETERS ==

# :param EMBEDDING_SIZES:
#       The embedding sizes D at which HDF and the folded Morgan fingerprints are compared.
EMBEDDING_SIZES: List[int] = [32, 64, 128, 256, 512, 1024, 2048]
# :param NUM_LAYERS:
#       Number of HDF message-passing steps.
NUM_LAYERS: int = 2
# :param MORGAN_RADIUS:
#       Radius of the Morgan fingerprints.
MORGAN_RADIUS: int = 2
# :param WL_ITERATIONS:
#       Number of 1-WL color refinement iterations.
WL_ITERATIONS: int = 2
# :param HYDROGEN_COUNT:
#       Hydrogen count of the HDF node attributes and of the 1-WL labels on the HDF graph: "total" counts all bonded
#       hydrogens (RDKit GetTotalNumHs), "implicit" only implicit ones (0 for bracket atoms such as [nH]).
HYDROGEN_COUNT: str = 'total'
# :param SPECTRUM:
#       Fourier magnitudes of the random HDF codebook vectors: "unit" (the HDF default since 2026-10-09) or
#       "gaussian" (the original encoder, used by the runs ex_16_collisions and ex_16_collisions_totalh).
SPECTRUM: str = 'unit'
# :param SEED:
#       Seed of the HDF codebooks, the atom permutations of the noise measurement and the random projections.
SEED: int = 0
# :param ENCODING_CHUNK_SIZE:
#       Molecules converted to graph dicts and encoded at once (bounds the memory of the graph dicts).
ENCODING_CHUNK_SIZE: int = 5000
# :param ENCODING_BATCH_SIZE:
#       Batch size of the HyperNet forward pass.
ENCODING_BATCH_SIZE: int = 1000

# == COLLISION PARAMETERS ==

# :param NOISE_MOLECULES:
#       Number of molecules whose atoms are permuted to measure the floating-point noise floor of HDF.
NOISE_MOLECULES: int = 500
# :param NOISE_PERMUTATIONS:
#       Random atom permutations per molecule for the noise measurement.
NOISE_PERMUTATIONS: int = 3
# :param TOLERANCE_FACTOR:
#       The HDF collision tolerance (Euclidean distance between unit vectors) is this factor times the largest
#       measured noise distance, but at least TOLERANCE_FLOOR.
TOLERANCE_FACTOR: float = 1000.0
# :param TOLERANCE_FLOOR:
#       Lower bound of the HDF collision tolerance.
TOLERANCE_FLOOR: float = 1e-12
# :param SEARCH_RADIUS:
#       All pairs of HDF vectors closer than this Euclidean distance are found and reported; those above the tolerance
#       are close but distinct and do not count as collisions. 1e-3 corresponds to a cosine similarity of 1 - 5e-7.
SEARCH_RADIUS: float = 1e-3
# :param NUM_PROJECTIONS:
#       Random projections for the pair search; the first one is sorted, the others prune the candidates.
NUM_PROJECTIONS: int = 4
# :param NUM_EXAMPLES:
#       Number of example groups (and close pairs) stored per representation.
NUM_EXAMPLES: int = 10

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# the element list of the continuous HDF node encoder; other elements share the "unknown" codebook entry
HDF_ELEMENTS = {int(z) for z in inspect.signature(make_molecule_node_encoder_map_cont).parameters['atoms'].default}


def wl_hash(mol: Chem.Mol, iterations: int, bond_labels: bool, hydrogens: str) -> str:
    """1-WL hash of the graph that HDF encodes (optionally with bond types as edge labels)."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        element = z if z in HDF_ELEMENTS else 'other'
        h = atom.GetTotalNumHs() if hydrogens == 'total' else atom.GetNumImplicitHs()
        graph.add_node(atom.GetIdx(), label=f'{element}|{h}|{atom.GetDegree()}')
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond=str(bond.GetBondType()))
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr='label', edge_attr='bond' if bond_labels else None,
                                           iterations=iterations)


def groups_from_keys(keys: list) -> List[List[int]]:
    """Groups (index lists) of molecules that share a key, only groups with at least two members."""
    members = defaultdict(list)
    for i, key in enumerate(keys):
        members[key].append(i)
    return [g for g in members.values() if len(g) > 1]


def groups_from_pairs(n: int, pairs_i: np.ndarray, pairs_j: np.ndarray) -> List[List[int]]:
    """Connected components (size >= 2) of the collision pairs, via union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs_i.tolist(), pairs_j.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    members = defaultdict(list)
    for i in set(pairs_i.tolist()) | set(pairs_j.tolist()):
        members[find(i)].append(i)
    return [sorted(g) for g in members.values() if len(g) > 1]


def summarize(groups: List[List[int]], n: int, smiles: List[str], num_examples: int) -> dict:
    sizes = sorted((len(g) for g in groups), reverse=True)
    largest = sorted(groups, key=len, reverse=True)[:num_examples]
    return {
        'collision_rate': sum(sizes) / n,
        'num_in_collisions': sum(sizes),
        'num_groups': len(groups),
        'largest_group': sizes[0] if sizes else 0,
        'examples': [[smiles[i] for i in g[:5]] for g in largest],
    }


def encode_hdf(net: HyperNet, mols: List[Chem.Mol], chunk_size: int, batch_size: int,
               hydrogens: str) -> np.ndarray:
    """Unit-normalized float64 HDF vectors; graph dicts are built per chunk to bound the memory."""
    out = np.empty((len(mols), net.hidden_dim), dtype=np.float64)
    for start in range(0, len(mols), chunk_size):
        graphs = []
        for mol in mols[start:start + chunk_size]:
            graph = graph_dict_from_mol(mol, hydrogens=hydrogens)
            graph.pop('graph_labels', None)
            graphs.append(graph)
        results = net.forward_graphs(graphs, batch_size=batch_size)
        out[start:start + len(graphs)] = np.stack([np.asarray(r['graph_embedding'], dtype=np.float64)
                                                  for r in results])
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def close_pairs(x: np.ndarray, radius: float, num_projections: int, seed: int, chunk: int = 5000):
    """All pairs (i < j in sorted order) with Euclidean distance <= radius, found exactly: a pair at distance d
    differs by at most d in each projection onto a unit vector, so it lies within the radius in all projections."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((x.shape[1], num_projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    projections = x @ directions
    order = np.argsort(projections[:, 0], kind='stable')
    p_sorted = projections[order]
    n = len(order)
    found_i, found_j, found_d = [], [], []
    num_candidates = 0
    for k in range(1, n):
        window = np.nonzero(p_sorted[k:, 0] - p_sorted[:n - k, 0] <= radius)[0]
        if len(window) == 0:
            break
        # prune with the other projections before computing full distances
        keep = np.all(np.abs(p_sorted[window + k, 1:] - p_sorted[window, 1:]) <= radius, axis=1)
        window = window[keep]
        num_candidates += len(window)
        for start in range(0, len(window), chunk):
            w = window[start:start + chunk]
            a, b = order[w], order[w + k]
            d = np.linalg.norm(x[a] - x[b], axis=1)
            mask = d <= radius
            found_i.append(a[mask])
            found_j.append(b[mask])
            found_d.append(d[mask])
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
    return cat(found_i, int), cat(found_j, int), cat(found_d, float), num_candidates


@experiment
def experiment(e: Experiment):

    if e.SPECTRUM not in ('gaussian', 'unit'):
        raise ValueError(f'SPECTRUM must be "gaussian" or "unit", got {e.SPECTRUM!r}')

    # --- distinct molecules ---
    e.log(f'loading dataset "{e.DATASET_NAME}"...')
    df = load_smiles_dataset(e.DATASET_NAME)
    raw = [str(s).strip() for s in df['smiles']]
    counts = Counter()
    mols, smiles, seen = [], [], set()
    for s in raw:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            counts['unparsable'] += 1
            continue
        if len(Chem.GetMolFrags(mol)) > 1:
            counts['multiple_fragments'] += 1
            continue
        # single heavy atoms have no edges; the prediction experiments drop them as well
        if mol.GetNumAtoms() < 2:
            counts['single_atom'] += 1
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        if canonical in seen:
            counts['duplicates'] += 1
            continue
        seen.add(canonical)
        smiles.append(canonical)
        mols.append(Chem.MolFromSmiles(canonical))
    if e.NUM_DATA is not None and e.NUM_DATA < len(mols):
        keep = sorted(random.Random(e.SEED).sample(range(len(mols)), e.NUM_DATA))
        mols, smiles = [mols[i] for i in keep], [smiles[i] for i in keep]
    n = len(mols)
    e['dataset/num_raw'] = len(raw)
    for key, value in counts.items():
        e[f'dataset/num_{key}'] = value
    e['dataset/num_molecules'] = n
    e.log(f'{n} distinct molecules ({dict(counts)})')

    # --- 1-WL ---
    wl_keys = {}
    for variant, bond_labels in (('hdf_graph', False), ('bond_orders', True)):
        start = time.time()
        wl_keys[variant] = [wl_hash(m, e.WL_ITERATIONS, bond_labels, e.HYDROGEN_COUNT) for m in mols]
        groups = groups_from_keys(wl_keys[variant])
        e[f'wl/{variant}/collisions'] = summarize(groups, n, smiles, e.NUM_EXAMPLES)
        e[f'wl/{variant}/time'] = time.time() - start
        e.log(f'1-WL ({variant}): collision rate {e[f"wl/{variant}/collisions"]["collision_rate"]:.5f}')

    # dataset statistics for the HDF graph-level encoders, as in the prediction experiments
    max_size = max(m.GetNumAtoms() for m in mols)
    max_diameter = max(int(np.max(Chem.GetDistanceMatrix(m))) for m in mols)
    e['dataset/max_size'] = max_size
    e['dataset/max_diameter'] = max_diameter
    e['dataset/mean_num_atoms'] = float(np.mean([m.GetNumAtoms() for m in mols]))

    for dim in e.EMBEDDING_SIZES:

        # --- Morgan ---
        start = time.time()
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=e.MORGAN_RADIUS, fpSize=dim)
        keys = [np.packbits(generator.GetFingerprintAsNumPy(m)).tobytes() for m in mols]
        e[f'morgan/{dim}/collisions'] = summarize(groups_from_keys(keys), n, smiles, e.NUM_EXAMPLES)
        e[f'morgan/{dim}/time'] = time.time() - start
        e.log(f'D={dim} Morgan: collision rate {e[f"morgan/{dim}/collisions"]["collision_rate"]:.5f}')

        # --- HDF ---
        net = HyperNet(
            hidden_dim=dim,
            depth=e.NUM_LAYERS,
            node_encoder_map=make_molecule_node_encoder_map_cont(dim=dim, seed=e.SEED,
                                                                 unit_modulus=(e.SPECTRUM == 'unit')),
            graph_encoder_map=make_molecule_graph_encoder_map_cont(dim=dim, seed=e.SEED, max_graph_size=max_size,
                                                                   max_graph_diameter=max_diameter,
                                                                   unit_modulus=(e.SPECTRUM == 'unit')),
            seed=e.SEED,
            normalize_all=True,
            bidirectional=True,
            device='cpu',
        )

        # floating-point noise floor (before the long encoding, so that a failure surfaces early): the same
        # molecules with randomly permuted atom orders, encoded in different batches
        rng = random.Random(e.SEED)
        permuted, owners = [], []
        for i in rng.sample(range(n), min(e.NOISE_MOLECULES, n)):
            for _ in range(e.NOISE_PERMUTATIONS):
                order = list(range(mols[i].GetNumAtoms()))
                rng.shuffle(order)
                mol = Chem.RenumberAtoms(mols[i], order)
                Chem.SanitizeMol(mol)
                permuted.append(mol)
                owners.append(i)
        reference = encode_hdf(net, [mols[i] for i in owners], e.ENCODING_CHUNK_SIZE, e.ENCODING_BATCH_SIZE,
                               e.HYDROGEN_COUNT)
        noise = np.linalg.norm(encode_hdf(net, permuted, e.ENCODING_CHUNK_SIZE, e.ENCODING_BATCH_SIZE,
                                          e.HYDROGEN_COUNT) - reference, axis=1)
        if not np.isfinite(noise).all():
            raise ValueError('non-finite HDF vectors in the noise measurement')
        tolerance = max(e.TOLERANCE_FLOOR, e.TOLERANCE_FACTOR * float(noise.max()))
        if tolerance >= e.SEARCH_RADIUS / 10:
            raise ValueError(f'collision tolerance {tolerance:.2e} is not well below the search radius')
        e[f'hdf/{dim}/noise_max'] = float(noise.max())
        e[f'hdf/{dim}/noise_median'] = float(np.median(noise))
        e[f'hdf/{dim}/tolerance'] = tolerance

        start = time.time()
        x = encode_hdf(net, mols, e.ENCODING_CHUNK_SIZE, e.ENCODING_BATCH_SIZE, e.HYDROGEN_COUNT)
        if not np.isfinite(x).all():
            raise ValueError('non-finite HDF vectors')
        e[f'hdf/{dim}/encode_time'] = time.time() - start

        # all pairs within the search radius; collisions are those within the tolerance
        start = time.time()
        pi, pj, pd, num_candidates = close_pairs(x, e.SEARCH_RADIUS, e.NUM_PROJECTIONS, e.SEED)
        e[f'hdf/{dim}/search_time'] = time.time() - start
        e[f'hdf/{dim}/num_candidates'] = num_candidates
        collision = pd <= tolerance
        groups = groups_from_pairs(n, pi[collision], pj[collision])
        e[f'hdf/{dim}/collisions'] = summarize(groups, n, smiles, e.NUM_EXAMPLES)
        close = ~collision
        e[f'hdf/{dim}/collision_max_distance'] = float(pd[collision].max()) if collision.any() else None
        e[f'hdf/{dim}/num_close_pairs'] = int(close.sum())
        e[f'hdf/{dim}/close_min_distance'] = float(pd[close].min()) if close.any() else None
        e[f'hdf/{dim}/close_examples'] = [[smiles[a], smiles[b], float(d)] for a, b, d in
                                          sorted(zip(pi[close], pj[close], pd[close]), key=lambda t: t[2])
                                          [:e.NUM_EXAMPLES]]

        # cross-tabulation with 1-WL on the HDF graph: HDF collision groups whose members 1-WL separates (HDF less
        # expressive than 1-WL there), and 1-WL collision groups whose members HDF separates (e.g. by the diameter)
        wl = wl_keys['hdf_graph']
        split_by_wl = [g for g in groups if len({wl[i] for i in g}) > 1]
        hdf_group_of = {i: k for k, g in enumerate(groups) for i in g}
        split_by_hdf = [g for g in groups_from_keys(wl) if len({hdf_group_of.get(i, ('alone', i)) for i in g}) > 1]
        e[f'hdf/{dim}/num_in_hdf_collisions_split_by_wl'] = sum(len(g) for g in split_by_wl)
        e[f'hdf/{dim}/hdf_collisions_split_by_wl_examples'] = [[smiles[i] for i in g[:5]]
                                                               for g in split_by_wl[:e.NUM_EXAMPLES]]
        e[f'hdf/{dim}/num_in_wl_collisions_split_by_hdf'] = sum(len(g) for g in split_by_hdf)
        e.log(f'D={dim} HDF: collision rate {e[f"hdf/{dim}/collisions"]["collision_rate"]:.5f}, '
              f'noise max {noise.max():.2e}, tolerance {tolerance:.2e}, close pairs {int(close.sum())}, '
              f'HDF collisions split by 1-WL {e[f"hdf/{dim}/num_in_hdf_collisions_split_by_wl"]}, '
              f'1-WL collisions split by HDF {e[f"hdf/{dim}/num_in_wl_collisions_split_by_hdf"]}, '
              f'encode {e[f"hdf/{dim}/encode_time"]:.0f} s, search {e[f"hdf/{dim}/search_time"]:.0f} s')


experiment.run_if_main()
