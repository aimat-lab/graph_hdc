"""
Exact graph edit distance between pairs of small molecules (ex_17, reviewer comment R2.7).

This experiment computes the exact graph edit distance (GED) for one chunk of a deterministic list of
molecule pairs. The GED values serve as the ground truth against which the distances of molecular
fingerprints are correlated in ``ged_exact_correlation.py``.

The GED is computed with the exact algorithm of NetworkX (``optimize_edit_paths``, the depth-first
branch-and-bound search behind ``graph_edit_distance``; Abu-Aisheh et al., ICPRAM 2015) on the heavy-atom
graphs of the molecules. Nodes are labeled with the element and edges with the RDKit bond type (single,
double, triple, aromatic). All six edit operations have unit cost, and substituting a node or an edge with
one of the same label costs nothing. Hydrogen atoms and formal charges are not part of the labels.

Two kinds of pair sets are supported (PAIR_SET):

- 'random': the first NUM_PAIRS pairs of distinct molecules of a seeded random stream. The GED is computed
  without an upper bound. A pair counts as finished, i.e. its GED as exact, if the search ends within
  TIMEOUT seconds; otherwise the recorded value is only the best upper bound found within that time.
- 'near': the first NUM_PAIRS pairs of an independent seeded random stream are screened with
  ``graph_edit_distance(..., upper_bound=UPPER_BOUND)``, which returns the exact GED if it is at most
  UPPER_BOUND and None otherwise. The near pairs are the screened pairs with a GED of at most UPPER_BOUND,
  so the near set contains all near pairs among the screened pairs.

The molecules are the unique canonical SMILES of all connected molecules of DATASET_NAME with at least 2
and at most MAX_HEAVY_ATOMS heavy atoms. The pair list depends only on DATASET_NAME, MAX_HEAVY_ATOMS,
PAIR_SET, NUM_PAIRS and PAIR_SEED, so the chunks (CHUNK_INDEX of NUM_CHUNKS) of one pair set can run as
independent jobs (see ``_slurm_ex_17.py``).

Output: ``pairs.csv`` in the archive folder with one row per pair, written as soon as the pair is done, so
partial results survive an interrupted run.
"""
import csv
import multiprocessing as mp
import os
import time
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The chem_mat_data dataset from which the molecules are drawn, e.g. 'qm9_smiles' or 'zinc250k'.
DATASET_NAME: str = 'qm9_smiles'

# :param MAX_HEAVY_ATOMS:
#       Only molecules with at most this many heavy atoms are used, which keeps the exact GED tractable.
#       QM9 contains at most 9 heavy atoms; for ZINC250k we use 12.
MAX_HEAVY_ATOMS: int = 9

# == PAIR PARAMETERS ==

# :param PAIR_SET:
#       'random' for randomly sampled pairs with an unbounded exact GED, or 'near' for randomly sampled pairs
#       that are screened for a GED of at most UPPER_BOUND (see the module docstring).
PAIR_SET: str = 'random'

# :param NUM_PAIRS:
#       The number of pairs in the full pair list of this pair set: the number of random pairs for 'random'
#       and the number of screened pairs for 'near'.
NUM_PAIRS: int = 10_000

# :param PAIR_SEED:
#       The seed of the random pair stream. The 'random' and 'near' sets use independent streams derived
#       from this seed.
PAIR_SEED: int = 0

# :param NUM_CHUNKS:
#       The number of chunks into which the pair list is split. Chunk k contains the contiguous block
#       [k * size, (k + 1) * size) of the pair list with size = ceil(NUM_PAIRS / NUM_CHUNKS).
NUM_CHUNKS: int = 1

# :param CHUNK_INDEX:
#       The index of the chunk that this run processes, between 0 and NUM_CHUNKS - 1.
CHUNK_INDEX: int = 0

# == GED PARAMETERS ==

# :param TIMEOUT:
#       The maximum computation time per pair in seconds. Pairs whose search does not end within this time
#       are marked as unfinished.
TIMEOUT: float = 600.0

# :param UPPER_BOUND:
#       The GED threshold of the 'near' pair set. Ignored for the 'random' pair set.
UPPER_BOUND: float = 4.0

# :param NUM_WORKERS:
#       The number of worker processes that compute the GED of different pairs in parallel. Each worker is
#       single-threaded, so this should match the number of CPUs of the job.
NUM_WORKERS: int = 4

# == EXPERIMENT PARAMETERS ==

__DEBUG__: bool = True
__TESTING__: bool = False

# The random streams of the two pair sets are kept independent by offsetting the seed.
stream_seed_offsets: dict = {'random': 0, 'near': 1}

csv_fields: List[str] = [
    'pair_index', 'smiles_1', 'smiles_2', 'n_atoms_1', 'n_atoms_2', 'n_bonds_1', 'n_bonds_2',
    'ged', 'finished', 'seconds', 'seconds_to_best', 'n_improvements',
]


def mol_to_graph(smiles: str) -> nx.Graph:
    """Heavy-atom graph with the element as node label and the RDKit bond type as edge label."""
    mol = Chem.MolFromSmiles(smiles)
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), el=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bo=str(bond.GetBondType()))
    return graph


def node_match(a: dict, b: dict) -> bool:
    return a['el'] == b['el']


def edge_match(a: dict, b: dict) -> bool:
    return a['bo'] == b['bo']


def load_molecule_pool(dataset_name: str, max_heavy_atoms: int) -> List[str]:
    """Sorted unique canonical SMILES of the connected molecules with 2..max_heavy_atoms heavy atoms."""
    from chem_mat_data import load_smiles_dataset
    df = load_smiles_dataset(dataset_name)
    column = 'smiles' if 'smiles' in df.columns else df.columns[0]
    pool = set()
    for smiles in df[column]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            continue
        if not 2 <= mol.GetNumHeavyAtoms() <= max_heavy_atoms:
            continue
        pool.add(Chem.MolToSmiles(mol))
    return sorted(pool)


def sample_pairs(num_molecules: int, num_pairs: int, seed: int) -> List[Tuple[int, int]]:
    """
    The first ``num_pairs`` distinct unordered pairs of distinct molecule indices of a seeded random stream.
    The result depends only on the three arguments.
    """
    if num_pairs > num_molecules * (num_molecules - 1) // 2:
        raise ValueError(f'cannot sample {num_pairs} distinct pairs from {num_molecules} molecules')
    rng = np.random.default_rng(seed)
    pairs: List[Tuple[int, int]] = []
    seen = set()
    while len(pairs) < num_pairs:
        batch = rng.integers(0, num_molecules, size=(num_pairs, 2))
        for i, j in batch:
            if i == j:
                continue
            key = (int(min(i, j)), int(max(i, j)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) == num_pairs:
                break
    return pairs


def compute_pair(args: tuple) -> dict:
    """Exact GED of one pair; with an upper bound, None means that the GED exceeds the bound."""
    pair_index, smiles_1, smiles_2, timeout, upper_bound = args
    RDLogger.DisableLog('rdApp.*')
    g1, g2 = mol_to_graph(smiles_1), mol_to_graph(smiles_2)
    start = time.perf_counter()
    best: Optional[float] = None
    seconds_to_best: Optional[float] = None
    n_improvements = 0
    if upper_bound is None:
        # strictly_decreasing=True yields only improving edit paths, exactly as graph_edit_distance does
        for _, _, cost in nx.optimize_edit_paths(g1, g2, node_match=node_match, edge_match=edge_match,
                                                 strictly_decreasing=True, timeout=timeout):
            best = cost
            seconds_to_best = time.perf_counter() - start
            n_improvements += 1
    else:
        best = nx.graph_edit_distance(g1, g2, node_match=node_match, edge_match=edge_match,
                                      upper_bound=upper_bound, timeout=timeout)
    seconds = time.perf_counter() - start
    return {
        'pair_index': pair_index,
        'smiles_1': smiles_1,
        'smiles_2': smiles_2,
        'n_atoms_1': g1.number_of_nodes(),
        'n_atoms_2': g2.number_of_nodes(),
        'n_bonds_1': g1.number_of_edges(),
        'n_bonds_2': g2.number_of_edges(),
        'ged': '' if best is None else best,
        # networkx prunes every branch once the elapsed time exceeds the limit, so a search that ends before
        # the limit has explored the whole search tree and its result is exact
        'finished': seconds < timeout,
        'seconds': round(seconds, 4),
        'seconds_to_best': '' if seconds_to_best is None else round(seconds_to_best, 4),
        'n_improvements': n_improvements,
    }


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):

    @e.testing
    def testing(e: Experiment):
        e.NUM_PAIRS = 8
        e.NUM_CHUNKS = 2
        e.CHUNK_INDEX = 0
        e.TIMEOUT = 20.0
        e.NUM_WORKERS = 2

    e.log_parameters()
    if e.PAIR_SET not in stream_seed_offsets:
        raise ValueError(f'unknown PAIR_SET "{e.PAIR_SET}", expected one of {list(stream_seed_offsets)}')
    if not 0 <= e.CHUNK_INDEX < e.NUM_CHUNKS:
        raise ValueError(f'CHUNK_INDEX {e.CHUNK_INDEX} outside of [0, {e.NUM_CHUNKS})')

    RDLogger.DisableLog('rdApp.*')
    e.log(f'loading the molecules of "{e.DATASET_NAME}" with at most {e.MAX_HEAVY_ATOMS} heavy atoms...')
    pool = load_molecule_pool(e.DATASET_NAME, e.MAX_HEAVY_ATOMS)
    e.log(f'molecule pool: {len(pool)} unique connected molecules')
    e['pool/num_molecules'] = len(pool)

    stream_seed = e.PAIR_SEED + stream_seed_offsets[e.PAIR_SET]
    pairs = sample_pairs(len(pool), e.NUM_PAIRS, stream_seed)
    chunk_size = -(-e.NUM_PAIRS // e.NUM_CHUNKS)
    begin = e.CHUNK_INDEX * chunk_size
    end = min(begin + chunk_size, e.NUM_PAIRS)
    upper_bound = e.UPPER_BOUND if e.PAIR_SET == 'near' else None
    tasks = [(index, pool[i], pool[j], e.TIMEOUT, upper_bound)
             for index, (i, j) in enumerate(pairs[begin:end], start=begin)]
    e.log(f'pair set "{e.PAIR_SET}" (stream seed {stream_seed}): chunk {e.CHUNK_INDEX} of {e.NUM_CHUNKS} '
          f'with pairs [{begin}, {end}) of {e.NUM_PAIRS}, upper bound {upper_bound}, '
          f'timeout {e.TIMEOUT:.0f} s, {e.NUM_WORKERS} workers')
    e['chunk/begin'] = begin
    e['chunk/end'] = end
    e['chunk/stream_seed'] = stream_seed

    time_start = time.time()
    num_done = num_finished = num_hits = 0
    seconds: List[float] = []
    with e.open('pairs.csv', mode='w') as file, mp.Pool(e.NUM_WORKERS) as pool_workers:
        writer = csv.DictWriter(file, fieldnames=csv_fields, lineterminator='\n')
        writer.writeheader()
        file.flush()
        for row in pool_workers.imap_unordered(compute_pair, tasks):
            writer.writerow(row)
            file.flush()
            num_done += 1
            num_finished += int(row['finished'])
            num_hits += int(row['ged'] != '')
            seconds.append(row['seconds'])
            if row['seconds'] > 60 or num_done % max(1, len(tasks) // 20) == 0 or num_done == len(tasks):
                e.log(f' * {num_done}/{len(tasks)} pairs done ({num_finished} finished) | '
                      f'pair {row["pair_index"]}: {row["n_atoms_1"]}x{row["n_atoms_2"]} atoms, '
                      f'GED {row["ged"]}, {row["seconds"]:.1f} s')

    duration = time.time() - time_start
    e['summary/num_pairs'] = num_done
    e['summary/num_finished'] = num_finished
    e['summary/num_with_ged'] = num_hits
    e['summary/seconds_total'] = float(np.sum(seconds)) if seconds else 0.0
    e['summary/seconds_median'] = float(np.median(seconds)) if seconds else 0.0
    e['summary/seconds_max'] = float(np.max(seconds)) if seconds else 0.0
    e['summary/duration'] = duration
    e.log(f'done: {num_done} pairs in {duration / 60:.1f} min, {num_finished} finished, '
          f'{num_hits} with a GED value' + (f' <= {upper_bound:g}' if upper_bound is not None else ''))


experiment.run_if_main()
