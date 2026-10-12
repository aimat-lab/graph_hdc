"""
Experiment 24 (ex_24): Bayesian optimization (BO) benchmark of the HDF revision (Reviewer 2, comment R2.4).

The original BO experiment (ex_07) used a Matern-5/2 GP on standardized inputs for every representation,
including the binary Morgan fingerprints, for which a Tanimoto kernel is the standard choice. This benchmark
repeats the size sweep of ex_07 with a kernel that suits each representation and with more objectives:

    objectives    clogp          ClogP (Crippen) closest to 15 on AqSolDB      (target, as in ex_07)
                  qed            QED closest to 0.2 on ZINC250k                (target, as in ex_07)
                  solubility     maximal AqSolDB solubility (logS)
                  lipophilicity  maximal lipophilicity (logD)
                  freesolv       minimal FreeSolv hydration free energy
                  beet           maximal honey-bee LD50 (BEET, exact values only)
    methods       hdf_matern       HDF (standard configuration) + Matern-5/2
                  morgan_tanimoto  Morgan radius 2, binary, folded to D bits + Tanimoto
                  morgan_pca       Morgan-2048 -> first D PCA components + Matern-5/2
                  random           random search
                  hdf_linear       HDF + linear (cosine) kernel, every objective except qed (SI)
    sizes         D = 8, 16, 32, 64, 128, 512
    campaigns     25 per (objective, method, size); campaign k shares its initial molecules across methods
                  and uses its own HDF codebook (seed 1000 + 100 k)
    budget        20 initial molecules + 150 rounds of one molecule; freesolv 20 + 80; beet 10 + 40. The
                  acquisition scores the whole library in every round, also on ZINC250k.

The BO procedure itself lives in ``graph_hdc.bo``. All data and results are written to ``_ex24/`` next to this
file (or to $EX24_ROOT): ``data/<library>/`` holds the cleaned library and its features, ``data/objectives/``
the objective values and initial designs, ``runs/`` one result file per campaign. Stages:

    python bo_benchmark.py featurize [--libraries ...] [--workers N]
    python bo_benchmark.py encode [--libraries ...] [--sizes ...] [--campaigns ...] [--workers N]
    python bo_benchmark.py tasks
    python bo_benchmark.py run --workers W --threads T [--gpus G] [--shard i --num-shards n]
    python bo_benchmark.py calibrate --workers-list 16 32 64 [--gpus G]
    python bo_benchmark.py status
    python bo_benchmark.py collect

On KCIST the stages run as whole-node SLURM jobs through ``run_ex24_kcist.sbatch``.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import socket
import subprocess
import sys
import time
import zlib
from typing import Dict, List, Optional

import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EX24_ROOT', os.path.join(PATH, '_ex24'))
DATA = os.path.join(ROOT, 'data')
RUNS = os.path.join(ROOT, 'runs')

SIZES = [8, 16, 32, 64, 128, 512]
NUM_CAMPAIGNS = 25
HDF_DEPTH = 2
MORGAN_RADIUS = 2
PCA_SOURCE_BITS = 2048
PCA_COMPONENTS = max(SIZES)
# Libraries with more molecules than this run their acquisition on a GPU (if the worker has one).
GPU_MIN_MOLECULES = 20000

# name -> chem_mat_data dataset, SMILES column and the property columns used as objectives
LIBRARIES = {
    'aqsoldb': {'smiles': 'SMILES', 'properties': {'solubility': 'Solubility'}},
    'zinc250k': {'smiles': 'smiles', 'properties': {'qed': 'qed'}},
    'freesolv': {'smiles': 'smiles', 'properties': {'freesolv': 'expt'}},
    'lipophilicity': {'smiles': 'smiles', 'properties': {'lipophilicity': 'exp'}},
    'beet': {'smiles': 'SMILES', 'properties': {'beet': 'Experimental value'}},
}

OBJECTIVES = {
    'clogp': dict(library='aqsoldb', property='clogp', mode='target', target=15.0, num_initial=20, num_rounds=150),
    'qed': dict(library='zinc250k', property='qed', mode='target', target=0.2, num_initial=20, num_rounds=150),
    'solubility': dict(library='aqsoldb', property='solubility', mode='maximize', target=None,
                       num_initial=20, num_rounds=150),
    'lipophilicity': dict(library='lipophilicity', property='lipophilicity', mode='maximize', target=None,
                          num_initial=20, num_rounds=150),
    'freesolv': dict(library='freesolv', property='freesolv', mode='minimize', target=None,
                     num_initial=20, num_rounds=80),
    'beet': dict(library='beet', property='beet', mode='maximize', target=None, num_initial=10, num_rounds=40),
}

GP_METHODS = ['hdf_matern', 'morgan_tanimoto', 'morgan_pca', 'hdf_linear']


def methods_for(objective: str) -> List[str]:
    """The methods of an objective: the linear-kernel HDF runs on every objective except ZINC250k (user)."""
    methods = ['hdf_matern', 'morgan_tanimoto', 'morgan_pca', 'random']
    if OBJECTIVES[objective]['library'] != 'zinc250k':
        methods.append('hdf_linear')
    return methods


def hdf_seed(campaign: int) -> int:
    """
    HDF codebook seed of a campaign. The encoder maps add offsets of up to +30 to their seed, so seeds 100
    apart never share a random stream between campaigns.
    """
    return 1000 + 100 * campaign


def campaign_seed(campaign: int) -> int:
    """Seed of the random tie-breaking and of the random search of a campaign (the same for all methods)."""
    return 24000 + campaign


def git_commit() -> str:
    try:
        return subprocess.run(['git', '-C', PATH, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return 'unknown'


def set_thread_env(threads: int) -> None:
    """
    Limits the thread pools of every numerical library for the worker processes spawned after this call (they
    inherit the environment and read it on import). Without it, numpy's OpenBLAS starts one thread per core in
    every worker, and dozens of workers per node spend their time in the kernel.
    """
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)


def log(message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def save_npy(path: str, array: np.ndarray) -> None:
    """Writes ``array`` atomically (temporary file + rename), so concurrent readers never see a partial file."""
    tmp = f'{path}.tmp{os.getpid()}.npy'
    np.save(tmp, array)
    os.replace(tmp, path)


# == FEATURIZATION ==

def _parse_row(args):
    """Worker of ``featurize``: one dataset row -> (stereo-free canonical SMILES, values) or None."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    smiles, values = args
    mol = Chem.MolFromSmiles(str(smiles))
    # The same filter as the original BO experiment: valid, at least two atoms and one bond, one component.
    # HDF needs the last two (single atoms have no messages, the graph diameter needs a connected graph).
    if mol is None or mol.GetNumAtoms() < 2 or mol.GetNumBonds() < 1 or len(Chem.GetMolFrags(mol)) > 1:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False), values


_GENERATORS = {}


def _featurize_molecule(smiles: str):
    """Worker of ``featurize``: graph dict, Morgan bits of every size and ClogP of one canonical SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, rdFingerprintGenerator
    from graph_hdc.special.molecules import graph_dict_from_mol
    RDLogger.DisableLog('rdApp.*')
    if not _GENERATORS:
        # binary Morgan fingerprints without chirality, folded to ``size`` bits (as in ex_07)
        for size in SIZES + [PCA_SOURCE_BITS]:
            _GENERATORS[size] = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=size)
    mol = Chem.MolFromSmiles(smiles)
    graph = graph_dict_from_mol(mol, hydrogens='total')
    graph.pop('graph_labels', None)
    bits = {size: generator.GetFingerprintAsNumPy(mol).astype(np.uint8) for size, generator in _GENERATORS.items()}
    return graph, bits, Crippen.MolLogP(mol)


def _parse_value(value) -> float:
    """Property value as float; NaN for missing, censored (">...") or non-numeric entries."""
    try:
        value = float(str(value).strip())
    except ValueError:
        return float('nan')
    return value if np.isfinite(value) else float('nan')


def featurize(library: str, workers: int) -> None:
    """
    Cleans one library and computes everything the BO campaigns need from it: graph dicts for HDF, Morgan bits
    of every size, the PCA of the 2048-bit Morgan fingerprints, property values, and for every objective on the
    library its objective values and the initial designs of all campaigns.
    """
    from chem_mat_data import load_smiles_dataset
    from sklearn.decomposition import PCA

    spec = LIBRARIES[library]
    folder = os.path.join(DATA, library)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(DATA, 'objectives'), exist_ok=True)

    set_thread_env(1)
    df = load_smiles_dataset(library)
    log(f'{library}: {len(df)} rows')
    names = list(spec['properties'])
    rows = [(smiles, [_parse_value(df[column].iloc[i]) for column in spec['properties'].values()])
            for i, smiles in enumerate(df[spec['smiles']])]
    with mp.get_context('spawn').Pool(workers) as pool:
        parsed = pool.map(_parse_row, rows, chunksize=500)

    # Molecules that only differ in stereochemistry are identical for both HDF and Morgan without chirality:
    # one candidate per stereo-free canonical SMILES, with the mean of its (finite) values.
    groups: Dict[str, List[List[float]]] = {}
    for item in parsed:
        if item is not None:
            groups.setdefault(item[0], []).append(item[1])
    smiles_list = sorted(groups)
    values = np.full((len(smiles_list), len(names)), np.nan)
    for i, smiles in enumerate(smiles_list):
        group = np.array(groups[smiles], dtype=np.float64)  # (num_duplicates, num_properties)
        for j in range(len(names)):
            finite = group[:, j][np.isfinite(group[:, j])]
            if len(finite) > 0:
                values[i, j] = finite.mean()
    # every candidate needs a value for every objective of the library (BEET: the censored rows have none)
    keep = np.isfinite(values).all(axis=1)
    num_invalid = sum(item is None for item in parsed)
    log(f'{library}: {len(parsed) - num_invalid} valid rows -> {len(smiles_list)} structures without stereo -> '
        f'{int(keep.sum())} candidates with values ({num_invalid} rows filtered as invalid, multi-component or '
        f'smaller than two atoms)')
    smiles_list = [s for s, k in zip(smiles_list, keep) if k]
    values = values[keep]

    with mp.get_context('spawn').Pool(workers) as pool:
        featurized = pool.map(_featurize_molecule, smiles_list, chunksize=200)
    graphs = [f[0] for f in featurized]
    clogp = np.array([f[2] for f in featurized], dtype=np.float64)
    for size in SIZES + [PCA_SOURCE_BITS]:
        save_npy(os.path.join(folder, f'morgan_{size}.npy'), np.stack([f[1][size] for f in featurized]))
    with open(os.path.join(folder, 'graphs.pkl'), 'wb') as file:
        pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)

    # PCA of the 2048-bit fingerprints, fitted once on the whole library (unsupervised). The components are
    # nested, so the first D columns are the D-dimensional projection for every size of the sweep. A library
    # with fewer molecules has fewer usable components (rank <= num_molecules - 1).
    bits = np.load(os.path.join(folder, f'morgan_{PCA_SOURCE_BITS}.npy')).astype(np.float32)
    num_components = min(PCA_COMPONENTS, len(smiles_list) - 1, PCA_SOURCE_BITS)
    pca = PCA(n_components=num_components, svd_solver='covariance_eigh' if len(bits) > PCA_SOURCE_BITS else 'full')
    scores = pca.fit_transform(bits).astype(np.float32)
    save_npy(os.path.join(folder, 'pca_scores.npy'), scores)
    save_npy(os.path.join(folder, 'pca_explained.npy'), pca.explained_variance_ratio_.astype(np.float64))

    with open(os.path.join(folder, 'library.tsv'), 'w') as file:
        file.write('\t'.join(['index', 'smiles', 'clogp'] + names) + '\n')
        for i, smiles in enumerate(smiles_list):
            file.write('\t'.join([str(i), smiles, f'{clogp[i]:.6g}'] + [f'{v:.6g}' for v in values[i]]) + '\n')

    meta = {
        'library': library,
        'num_rows': len(df),
        'num_candidates': len(smiles_list),
        'num_filtered': num_invalid,
        'max_graph_size': int(max(g['graph_size'] for g in graphs)),
        'max_graph_diameter': int(max(g['graph_diameter'] for g in graphs)),
        'pca_components': int(num_components),
        'pca_explained_by_size': {str(s): float(pca.explained_variance_ratio_[:s].sum())
                                  for s in SIZES if s <= num_components},
        'git': git_commit(),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(folder, 'meta.json'), 'w') as file:
        json.dump(meta, file, indent=2)

    from graph_hdc.bo import objective_values
    properties = dict(zip(names, values.T))
    properties['clogp'] = clogp
    for objective, ospec in OBJECTIVES.items():
        if ospec['library'] != library:
            continue
        objective_array = objective_values(properties[ospec['property']], ospec['mode'], ospec['target'])
        initial = np.stack([
            np.random.default_rng([24, zlib.crc32(objective.encode()), k]).choice(
                len(smiles_list), size=ospec['num_initial'], replace=False)
            for k in range(NUM_CAMPAIGNS)
        ])
        np.savez(os.path.join(DATA, 'objectives', f'{objective}.npz'),
                 values=properties[ospec['property']], objective=objective_array, initial=initial,
                 objective_max=objective_array.max(), objective_min=objective_array.min())
        log(f'{objective}: {len(smiles_list)} candidates, objective range [{objective_array.min():.4g}, '
            f'{objective_array.max():.4g}], best molecule {smiles_list[int(objective_array.argmax())]}')
    log(f'{library}: done, {json.dumps({k: meta[k] for k in ("max_graph_size", "max_graph_diameter")})}')


# == HDF ENCODING ==

_ENCODE_STATE = {}


def _encode_init(library: str, threads: int):
    import torch
    from graph_hdc.graph import data_list_from_graph_dicts
    torch.set_num_threads(threads)
    with open(os.path.join(DATA, library, 'graphs.pkl'), 'rb') as file:
        graphs = pickle.load(file)
    with open(os.path.join(DATA, library, 'meta.json')) as file:
        meta = json.load(file)
    _ENCODE_STATE.update(library=library, data=data_list_from_graph_dicts(graphs), meta=meta)


def encode_hdf(data_list, size: int, seed: int, max_size: float, max_diameter: float,
               batch_size: int = 1024) -> np.ndarray:
    """
    HDF embeddings in the standard configuration (bidirectional message passing, unit-modulus codebooks, total
    hydrogen counts in the graph dicts, size and diameter encodings, depth 2) of the PyG ``data_list``.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from graph_hdc.models import HyperNet
    from graph_hdc.special.molecules import make_molecule_graph_encoder_map_cont, make_molecule_node_encoder_map_cont

    net = HyperNet(
        hidden_dim=size,
        depth=HDF_DEPTH,
        node_encoder_map=make_molecule_node_encoder_map_cont(dim=size, seed=seed),
        graph_encoder_map=make_molecule_graph_encoder_map_cont(dim=size, seed=seed, max_graph_size=max_size,
                                                               max_graph_diameter=max_diameter),
        seed=seed,
        normalize_all=True,
        bidirectional=True,
    )
    embeddings = []
    with torch.no_grad():
        for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
            embeddings.append(net.forward(batch)['graph_embedding'].to(torch.float32).cpu().numpy())
    return np.concatenate(embeddings)


def _encode_one(job):
    size, campaign = job
    library, meta = _ENCODE_STATE['library'], _ENCODE_STATE['meta']
    path = os.path.join(DATA, library, f'hdf_{size}_c{campaign}.npy')
    if os.path.exists(path):
        return path, 0.0
    start = time.perf_counter()
    embeddings = encode_hdf(_ENCODE_STATE['data'], size, hdf_seed(campaign), meta['max_graph_size'],
                            meta['max_graph_diameter'])
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f'non-finite HDF embeddings for {library} D={size} campaign {campaign}')
    save_npy(path, embeddings)
    return path, time.perf_counter() - start


def encode(library: str, sizes: List[int], campaigns: List[int], workers: int, threads: int) -> None:
    jobs = [(size, campaign) for size in sorted(sizes, reverse=True) for campaign in campaigns
            if not os.path.exists(os.path.join(DATA, library, f'hdf_{size}_c{campaign}.npy'))]
    log(f'{library}: {len(jobs)} HDF encodings to compute with {workers} workers x {threads} threads')
    if not jobs:
        return
    set_thread_env(threads)
    with mp.get_context('spawn').Pool(min(workers, len(jobs)), initializer=_encode_init,
                                      initargs=(library, threads)) as pool:
        for i, (path, seconds) in enumerate(pool.imap_unordered(_encode_one, jobs), start=1):
            log(f'  [{i}/{len(jobs)}] {os.path.basename(path)} in {seconds:.1f} s')


# == TASKS ==

def result_path(task: dict) -> str:
    size = 'na' if task['size'] is None else task['size']
    return os.path.join(RUNS, task['objective'], task['method'], f'D{size}', f'c{task["campaign"]}.npz')


def library_meta(library: str) -> dict:
    with open(os.path.join(DATA, library, 'meta.json')) as file:
        return json.load(file)


def estimated_cost(task: dict, num_molecules: int) -> float:
    """Rough relative cost of a campaign (for longest-first scheduling): GP fits plus library scans."""
    if task['method'] == 'random':
        return 0.001
    rounds = OBJECTIVES[task['objective']]['num_rounds']
    scan = num_molecules * task['size'] / 2e7
    if num_molecules > GPU_MIN_MOLECULES:
        scan /= 20.0  # GPU acquisition
    return rounds * (0.5 + scan)


def make_tasks(objectives: Optional[List[str]] = None, sizes: Optional[List[int]] = None,
               campaigns: Optional[List[int]] = None) -> List[dict]:
    """All campaigns of the benchmark (or of a subset), the most expensive first."""
    tasks = []
    for objective, spec in OBJECTIVES.items():
        if objectives is not None and objective not in objectives:
            continue
        meta = library_meta(spec['library'])
        for method in methods_for(objective):
            for size in [None] if method == 'random' else (sizes or SIZES):
                # a PCA cannot have more components than the library allows (BEET: 127)
                if method == 'morgan_pca' and size > meta['pca_components']:
                    continue
                for campaign in (campaigns if campaigns is not None else range(NUM_CAMPAIGNS)):
                    task = dict(objective=objective, method=method, size=size, campaign=campaign)
                    task['cost'] = estimated_cost(task, meta['num_candidates'])
                    tasks.append(task)
    tasks.sort(key=lambda t: -t['cost'])
    return tasks


# == RUNNING CAMPAIGNS ==

_WORKER = {}


def _worker_init(threads: int, gpus: int):
    import torch
    os.environ['OMP_NUM_THREADS'] = str(threads)
    torch.set_num_threads(threads)
    # pool workers are numbered 1, 2, ... (also replacements of finished workers): spread them over the GPUs
    identity = mp.current_process()._identity
    _WORKER['gpu'] = (identity[0] - 1) % gpus if gpus > 0 and identity else None
    _WORKER['threads'] = threads
    _WORKER['features'] = {}


def load_features(objective: str, method: str, size: Optional[int], campaign: int) -> Optional[np.ndarray]:
    """GP inputs of a campaign, in the form the method's kernel expects (see graph_hdc.bo)."""
    from graph_hdc.bo import center_scale
    library = OBJECTIVES[objective]['library']
    folder = os.path.join(DATA, library)
    if method == 'random':
        return None
    key = (library, method, size, campaign if method.startswith('hdf') else None)
    cache = _WORKER.setdefault('features', {})
    if key in cache:
        return cache[key]
    if method == 'hdf_matern':
        features = center_scale(np.load(os.path.join(folder, f'hdf_{size}_c{campaign}.npy')))
    elif method == 'hdf_linear':
        features = np.load(os.path.join(folder, f'hdf_{size}_c{campaign}.npy'))
        features = (features / np.linalg.norm(features, axis=1, keepdims=True)).astype(np.float32)
    elif method == 'morgan_tanimoto':
        features = np.load(os.path.join(folder, f'morgan_{size}.npy')).astype(np.float32)
    elif method == 'morgan_pca':
        features = center_scale(np.load(os.path.join(folder, 'pca_scores.npy'), mmap_mode='r')[:, :size])
    else:
        raise ValueError(method)
    # keep only the deterministic (campaign-independent) features of the current library; HDF features change
    # with every campaign and are never reused
    if not method.startswith('hdf'):
        cache.clear()
        cache[key] = features
    return features


def run_task(task: dict) -> dict:
    import torch
    from graph_hdc.bo import run_campaign
    path = result_path(task)
    if os.path.exists(path):
        return {**task, 'status': 'exists'}
    spec = OBJECTIVES[task['objective']]
    data = np.load(os.path.join(DATA, 'objectives', f'{task["objective"]}.npz'))
    num_molecules = len(data['objective'])
    gpu = _WORKER.get('gpu')
    device = f'cuda:{gpu}' if gpu is not None and num_molecules > GPU_MIN_MOLECULES else 'cpu'
    start = time.time()
    features = load_features(task['objective'], task['method'], task['size'], task['campaign'])
    time_features = time.time() - start
    result = run_campaign(
        method=task['method'],
        objective=data['objective'],
        initial=data['initial'][task['campaign']],
        num_rounds=spec['num_rounds'],
        seed=campaign_seed(task['campaign']),
        features=features,
        device=device,
    )
    meta = {
        **{k: task[k] for k in ('objective', 'method', 'size', 'campaign')},
        'hdf_seed': hdf_seed(task['campaign']) if task['method'].startswith('hdf') else None,
        'campaign_seed': campaign_seed(task['campaign']),
        'num_initial': spec['num_initial'], 'num_rounds': spec['num_rounds'],
        'num_molecules': num_molecules,
        'objective_max': float(data['objective_max']), 'objective_min': float(data['objective_min']),
        'device': device, 'threads': _WORKER.get('threads'), 'host': socket.gethostname(),
        'git': _WORKER.setdefault('git', git_commit()), 'torch': torch.__version__,
        'seconds_features': time_features, 'seconds_total': time.time() - start,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}.npz'
    np.savez(tmp, meta=json.dumps(meta), **result)
    os.replace(tmp, path)
    return {**task, 'status': 'done', 'seconds': meta['seconds_total'], 'device': device}


def run(workers: int, threads: int, gpus: int, shard: int, num_shards: int, limit: Optional[int] = None) -> None:
    with open(os.path.join(ROOT, 'tasks.jsonl')) as file:
        tasks = [json.loads(line) for line in file]
    # round-robin over the cost-sorted list gives every shard the same mix of long and short campaigns
    tasks = [t for i, t in enumerate(tasks) if i % num_shards == shard]
    tasks = [t for t in tasks if not os.path.exists(result_path(t))]
    if limit is not None:
        tasks = tasks[:limit]
    log(f'shard {shard}/{num_shards}: {len(tasks)} open campaigns, {workers} workers x {threads} threads, '
        f'{gpus} GPUs, host {socket.gethostname()}, code {git_commit()}')
    if not tasks:
        return
    set_thread_env(threads)
    context = mp.get_context('spawn')
    failed = 0
    start = time.time()
    with context.Pool(workers, initializer=_worker_init, initargs=(threads, gpus)) as pool:
        results = pool.imap_unordered(_run_task_safe, tasks, chunksize=1)
        for i, result in enumerate(results, start=1):
            if result['status'] == 'error':
                failed += 1
                log(f'  !!! {result["objective"]} {result["method"]} D={result["size"]} c{result["campaign"]}: '
                    f'{result["error"]}')
            if i % 50 == 0 or i == len(tasks):
                log(f'  {i}/{len(tasks)} campaigns, {failed} failed, {(time.time() - start) / 60:.1f} min')
    log(f'shard {shard}: finished, {failed} failed')


def _run_task_safe(task: dict) -> dict:
    import traceback
    try:
        return run_task(task)
    except Exception as error:
        return {**task, 'status': 'error', 'error': f'{error!r}\n{traceback.format_exc()}'}


# == CALIBRATION ==

def _calibrate_task(job):
    """Times ``rounds`` BO rounds of one configuration at the full-budget number of observations."""
    import torch
    from graph_hdc.bo import run_campaign
    objective, method, size, rounds, device = job
    data = np.load(os.path.join(DATA, 'objectives', f'{objective}.npz'))
    spec = OBJECTIVES[objective]
    # start with the number of observations of the last rounds of a real campaign
    num_observed = spec['num_initial'] + spec['num_rounds'] - rounds
    initial = np.random.default_rng(0).choice(len(data['objective']), size=num_observed, replace=False)
    features = load_features(objective, method, size, 0)
    if device != 'cpu' and _WORKER.get('gpu') is not None:
        device = f'cuda:{_WORKER["gpu"]}'
    elif device != 'cpu':
        device = 'cpu'
    start = time.perf_counter()
    result = run_campaign(method, data['objective'], initial, rounds, seed=0, features=features, device=device)
    return {'objective': objective, 'method': method, 'size': size, 'device': device,
            'threads': _WORKER['threads'], 'seconds': time.perf_counter() - start,
            'fit': float(np.median(result['fit_seconds'])), 'acq': float(np.median(result['acq_seconds']))}


def calibrate(workers_list: List[int], threads_list: List[int], gpus: int, rounds: int) -> None:
    """
    Two measurements for planning the full run on a node: (1) single-worker seconds per round of every GP
    method on the largest and a mid-sized library at several sizes and thread counts, and (2) the throughput of
    the node (BO rounds per second) with W concurrent workers on a representative mix of campaigns.
    """
    report = {'host': socket.gethostname(), 'code': git_commit(), 'single': [], 'throughput': []}
    context = mp.get_context('spawn')

    # (1) single-worker timings
    for threads in threads_list:
        # ZINC250k on the GPU at every size (and on the CPU at one size for comparison), AqSolDB on the CPU
        jobs = [(objective, method, size, rounds, device)
                for method in ('hdf_matern', 'morgan_tanimoto', 'morgan_pca')
                for objective, size, device in
                [('solubility', s, 'cpu') for s in (8, 64, 512)] + [('qed', 64, 'cpu')] +
                ([('qed', s, 'cuda') for s in (8, 64, 512)] if gpus > 0 else [])]
        set_thread_env(threads)
        with context.Pool(1, initializer=_worker_init, initargs=(threads, gpus)) as pool:
            for timing in pool.imap(_calibrate_task, jobs):
                report['single'].append(timing)
                log(f'  single: {json.dumps(timing)}')

    # (2) throughput with W concurrent workers: every worker runs the same mix of small-library campaigns
    # (CPU) and ZINC campaigns (GPU acquisition)
    mix = [('solubility', 'hdf_matern', 64, rounds, 'cpu'), ('lipophilicity', 'morgan_tanimoto', 512, rounds, 'cpu'),
           ('freesolv', 'morgan_pca', 128, rounds, 'cpu'), ('qed', 'morgan_tanimoto', 64, rounds, 'cuda')]
    for workers in workers_list:
        for threads in threads_list:
            if workers * threads > (os.cpu_count() or 1):
                continue
            jobs = [mix[i % len(mix)] for i in range(2 * workers)]
            set_thread_env(threads)
            start = time.perf_counter()
            with context.Pool(workers, initializer=_worker_init, initargs=(threads, gpus)) as pool:
                timings = list(pool.imap_unordered(_calibrate_task, jobs))
            seconds = time.perf_counter() - start
            entry = {'workers': workers, 'threads': threads, 'jobs': len(jobs), 'seconds': seconds,
                     'rounds_per_second': len(jobs) * rounds / seconds,
                     'mean_job_seconds': float(np.mean([t['seconds'] for t in timings]))}
            report['throughput'].append(entry)
            log(f'  throughput: {json.dumps(entry)}')

    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, f'calibration_{socket.gethostname()}_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(path, 'w') as file:
        json.dump(report, file, indent=2)
    log(f'calibration report: {path}')


# == STATUS & COLLECTION ==

def status() -> None:
    with open(os.path.join(ROOT, 'tasks.jsonl')) as file:
        tasks = [json.loads(line) for line in file]
    counts: Dict[tuple, List[int]] = {}
    for task in tasks:
        key = (task['objective'], task['method'])
        done, total = counts.get(key, [0, 0])
        counts[key] = [done + os.path.exists(result_path(task)), total + 1]
    for (objective, method), (done, total) in sorted(counts.items()):
        print(f'{objective:15s} {method:16s} {done:5d}/{total}')
    print(f'total: {sum(c[0] for c in counts.values())}/{len(tasks)}')


def collect() -> None:
    """One row per finished campaign with its metrics (normalized-regret AUC, final regret, hit) and timings."""
    from graph_hdc.bo import normalized_regret, regret_auc
    with open(os.path.join(ROOT, 'tasks.jsonl')) as file:
        tasks = [json.loads(line) for line in file]
    rows = []
    for task in tasks:
        path = result_path(task)
        if not os.path.exists(path):
            continue
        result = np.load(path)
        meta = json.loads(str(result['meta']))
        regret = normalized_regret(result['best'], meta['objective_max'], meta['objective_min'])
        rows.append({
            **{k: meta[k] for k in ('objective', 'method', 'size', 'campaign')},
            'auc': regret_auc(regret), 'final_regret': float(regret[-1]), 'hit': bool(regret.min() <= 0.05),
            'first_hit': int(np.argmax(regret <= 0.05)) if (regret <= 0.05).any() else -1,
            'fit_seconds': float(result['fit_seconds'].sum()), 'acq_seconds': float(result['acq_seconds'].sum()),
            'fit_failures': int(result['fit_failures']), 'device': meta['device'],
            'seconds_total': meta['seconds_total'], 'git': meta['git'],
        })
    path = os.path.join(ROOT, 'campaigns.tsv')
    with open(path, 'w') as file:
        file.write('\t'.join(rows[0]) + '\n')
        for row in rows:
            file.write('\t'.join(str(v) for v in row.values()) + '\n')
    log(f'{len(rows)}/{len(tasks)} campaigns -> {path}')


# == COMMAND LINE ==

def parse_campaigns(text: str) -> List[int]:
    if '-' in text:
        first, last = text.split('-')
        return list(range(int(first), int(last) + 1))
    return [int(c) for c in text.split(',')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('featurize')
    p.add_argument('--libraries', nargs='+', default=list(LIBRARIES))
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('encode')
    p.add_argument('--libraries', nargs='+', default=list(LIBRARIES))
    p.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    p.add_argument('--campaigns', default=f'0-{NUM_CAMPAIGNS - 1}')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=1)
    p = sub.add_parser('tasks')
    p.add_argument('--objectives', nargs='+', default=None)
    p.add_argument('--sizes', nargs='+', type=int, default=None)
    p.add_argument('--campaigns', default=None)
    p = sub.add_parser('run')
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=None)
    p = sub.add_parser('calibrate')
    p.add_argument('--workers-list', nargs='+', type=int, default=[16, 32, 64])
    p.add_argument('--threads-list', nargs='+', type=int, default=[1, 2])
    p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--rounds', type=int, default=5)
    sub.add_parser('status')
    sub.add_parser('collect')
    args = parser.parse_args(argv)

    if args.stage == 'featurize':
        for library in args.libraries:
            featurize(library, args.workers)
    elif args.stage == 'encode':
        for library in args.libraries:
            encode(library, args.sizes, parse_campaigns(args.campaigns), args.workers, args.threads)
    elif args.stage == 'tasks':
        tasks = make_tasks(args.objectives, args.sizes,
                           parse_campaigns(args.campaigns) if args.campaigns else None)
        os.makedirs(ROOT, exist_ok=True)
        with open(os.path.join(ROOT, 'tasks.jsonl'), 'w') as file:
            for task in tasks:
                file.write(json.dumps(task) + '\n')
        log(f'{len(tasks)} campaigns -> {os.path.join(ROOT, "tasks.jsonl")}')
    elif args.stage == 'run':
        run(args.workers, args.threads, args.gpus, args.shard, args.num_shards, args.limit)
    elif args.stage == 'calibrate':
        calibrate(args.workers_list, args.threads_list, args.gpus, args.rounds)
    elif args.stage == 'status':
        status()
    elif args.stage == 'collect':
        collect()


if __name__ == '__main__':
    sys.exit(main())
