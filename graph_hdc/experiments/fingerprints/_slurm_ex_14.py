"""
Command-file generator for Experiment 14 (ex_14): comparison of HDF with graph neural networks.

Answers reviewer comment R2.1 ("missing GNN baseline"): do the gains of hyperdimensional fingerprints come
from the HDC algebra or simply from multi-hop message passing? Every one of the three standard PyG GNNs
(GCN, GIN, GATv2) is evaluated in two ways and compared against HDF with the same downstream models:

* ``trained``  the GNN trained end-to-end on the target (fixed standard config: 3x128 conv, dense head
               128-64-32, lr 1e-4, batch 32), with validation-based best-checkpoint restoration and early
               stopping (patience 50, max 1000 epochs).
* ``random``   the same architecture with frozen random weights (width 2048 = HDF dimension, 2 layers =
               HDF depth, sum pooling) as a training-free encoder, followed by the MLP and KNN.
* ``hdf``      HDF (D=2048, L=2; the ex_13 fixed config) followed by the same MLP and KNN.

All GNNs receive exactly the atom information HDF encodes (NODE_FEATURES='hdf'). The MLP is the ex_13
fixed-table MLP ((100, 100), lr 1e-3); KNN uses k=5 with Euclidean distance for every representation.

Stages / command files written to ``_ex14/`` (executed by ``run_ex14_kcist.sbatch``):

* ``warmup.txt``  one cheap run per DATASET_NAME -> builds the shared ``load__``/``stats_`` caches.
                  Must finish before anything else starts (the caches are shared across all runs).
* ``hdf.txt``     all HDF runs, "primer" lines first: HDF encodings are cached per (DATASET_NAME, SEED),
                  so targets that share a dataset (the three QM9 targets) would race on the cache file.
                  The sbatch script runs the primer block to completion before the rest.
* ``gnn.txt``     all trained and random GNN runs (no shared caches besides ``load__``), sorted by expected
                  cost (longest first). With the round-robin sharding over the array tasks, every shard
                  then gets the same number (+-1) of runs of each cost class, and each node's work queue
                  starts its longest runs first.
* ``smoke_*``     the same pipeline on FreeSolv with seed 0, all 7 variants, few epochs.

* ``missing_*``   written by the ``missing`` mode: the lines of hdf.txt / gnn.txt that have no completed
                  (status 'done') archive yet, e.g. after a timeout, a node failure or failed runs. Submit
                  them with HDF_FILE / GNN_FILE pointing to these files (see run_ex14_kcist.sbatch).

Usage:
    python _slurm_ex_14.py            # writes the full command files
    python _slurm_ex_14.py smoke      # writes the smoke command files only
    python _slurm_ex_14.py missing    # writes missing_hdf.txt / missing_gnn.txt (no jobs may be running!)
    python _slurm_ex_14.py missing smoke  # the same for the smoke files / ex_14_smoke prefix
"""
import os
import re
import sys
import glob
import json

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex14')

PREFIX = 'ex_14_gnn'
PREFIX_WARMUP = 'ex_14_warmup'
SEEDS = list(range(10))
WARMUP_SEED = 420

# (note, DATASET_NAME, TARGET_INDEX, extra_params, big) -- a representative subset of the ex_13 targets
# (same tuples as _slurm_ex_13.DATASETS): small experimental datasets and the large quantum-chemistry
# datasets, with intensive (gap) and extensive (U0, ZPVE) properties. ClogP is left out on purpose: it is
# a sum of per-atom Crippen contributions, which a sum-pooling GNN can reproduce trivially.
DATASETS = [
    ('freesolv_hfe',       'freesolv',     0,    {}, False),
    ('aqsoldb_logs',       'aqsoldb',      None, {}, False),
    ('lipophilicity_logD', 'lipophilicity', 0,   {}, False),
    ('bace_ic50',          'bace_reg',     0,    {}, False),
    ('hopv15_pce',         'hopv15_exp',   5,    {}, False),
    ('compas_gap',         'compas_3x',    2,    {}, True),
    ('qm9_gap',            'qm9_smiles',   7,    {}, True),
    ('qm9_energy',         'qm9_smiles',   10,   {}, True),
    ('qm9_zpve',           'qm9_smiles',   9,    {}, True),
]

ARCHS = ['gcn', 'gin', 'gatv2']

# Downstream models on top of the fixed representations (HDF and random GNN).
DOWNSTREAM = {
    'MODELS': ['neural_net2', 'k_neighbors'],
    'NN_HIDDEN_LAYER_SIZES': (100, 100),
    'NN_LEARNING_RATE_INIT': 0.001,
    'KN_NUM_NEIGHBORS': 5,
    'KN_WEIGHTS': 'uniform',
    'KN_METRIC': 'minkowski',
}

VARIANTS = {
    # module suffix, params (MODELS for the trained GNN is filled in per architecture)
    'hdf': ('hdc', {
        'EMBEDDING_SIZE': 2048, 'NUM_LAYERS': 2, 'ENCODING_MODE': 'continuous', 'DEVICE': 'cpu',
        **DOWNSTREAM,
    }),
    'random': ('gnn_random', {
        'EMBEDDING_SIZE': 2048, 'NUM_LAYERS': 2, 'NODE_FEATURES': 'hdf', 'DEVICE': 'cuda',
        **DOWNSTREAM,
    }),
    'trained': ('gnn', {
        'NODE_FEATURES': 'hdf', 'CONV_UNITS': [128, 128, 128], 'DENSE_UNITS': [128, 64, 32],
        'BATCH_SIZE': 32, 'LEARNING_RATE': 1e-4, 'EPOCHS': 1000, 'EARLY_STOPPING_PATIENCE': 50,
    }),
}


def _command(module: str, prefix: str, seed: int, ds: tuple, params: dict) -> str:
    """One ``python predict_molecules__<module>.py --K="repr(v)" ...`` line (same format as ex_13)."""
    note, name, tidx, extra, _big = ds
    full = {
        '__DEBUG__': False, '__PREFIX__': prefix, 'SEED': seed,
        'NUM_TEST': 0.1, 'NUM_TRAIN': 1.0, 'NUM_DATA': 1.0,
        'DATASET_NAME': name, 'DATASET_TYPE': 'regression', 'NOTE': note,
        'NN_NUM_WORKERS': 0,
    }
    if tidx is not None:
        full['TARGET_INDEX'] = tidx
    full.update(params)
    full.update(extra)
    # Relative script path: the command files are executed from this folder (see run_ex14_kcist.sbatch),
    # so the same files work locally and on the cluster.
    parts = ['python', f'predict_molecules__{module}.py']
    parts += [f'--{k}="{v!r}"' for k, v in full.items()]
    return ' '.join(parts)


def variant_commands(ds: tuple, seed: int, prefix: str, overrides: dict = {}) -> dict:
    """All commands for one (dataset, seed), keyed by variant name ('hdf', 'random_gcn', 'trained_gin', ...)."""
    cmds = {}
    module, params = VARIANTS['hdf']
    cmds['hdf'] = _command(module, prefix, seed, ds, {**params, **overrides.get('hdf', {})})
    for arch in ARCHS:
        module, params = VARIANTS['random']
        cmds[f'random_{arch}'] = _command(module, prefix, seed, ds,
                                          {**params, 'GNN_ARCH': arch, **overrides.get('random', {})})
        module, params = VARIANTS['trained']
        cmds[f'trained_{arch}'] = _command(module, prefix, seed, ds,
                                           {**params, 'MODELS': [arch], **overrides.get('trained', {})})
    return cmds


def build_warmup(datasets: list) -> list:
    """One cheap HDC run (small D, linear model) per unique DATASET_NAME builds the load__/stats_ caches."""
    seen, lines = set(), []
    for ds in datasets:
        if ds[1] in seen:
            continue
        seen.add(ds[1])
        lines.append(_command('hdc', PREFIX_WARMUP, WARMUP_SEED, ds, {
            'EMBEDDING_SIZE': 256, 'NUM_LAYERS': 1, 'DEVICE': 'cpu', 'MODELS': ['linear'],
        }))
    return lines


def build(datasets: list, seeds: list, prefix: str, overrides: dict = {}) -> tuple:
    hdf_primer, hdf_rest, gnn = [], [], []
    primed = set()
    for seed in seeds:
        for ds in datasets:
            cmds = variant_commands(ds, seed, prefix, overrides)
            key = (ds[1], seed)
            (hdf_rest if key in primed else hdf_primer).append(cmds.pop('hdf'))
            primed.add(key)
            gnn += [(expected_cost(name, ds), line) for name, line in cmds.items()]
    # stable sort: within a cost class the seed-major, dataset-minor order is kept
    gnn = [line for _, line in sorted(gnn, key=lambda t: -t[0])]
    return hdf_primer, hdf_rest, gnn


def expected_cost(variant: str, ds: tuple) -> int:
    """
    Cost class of a run, used to balance the array shards (higher = longer). Every (variant, dataset
    family) combination is its own class, so that the round-robin sharding splits each class evenly;
    e.g. a trained QM9 run costs ~3.4x a trained COMPAS run and must not share a class with it.
    """
    trained = variant.startswith('trained')
    if ds[1] == 'qm9_smiles':
        return 5 if trained else 3
    if ds[1] == 'compas_3x':
        return 4 if trained else 2
    return 1 if trained else 0


def _write(name: str, lines: list):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'wrote {len(lines):4d} commands -> _ex14/{name}')


def identity_of_line(line: str) -> tuple:
    """(module, NOTE, SEED, arch) of a command line; arch is GNN_ARCH or MODELS[0] ('' for HDF)."""
    module = re.search(r'predict_molecules__(\w+)\.py', line).group(1)
    params = dict(re.findall(r'--(\w+)="([^"]*)"', line))
    note, seed = eval(params['NOTE']), eval(params['SEED'])
    if module == 'gnn_random':
        return module, note, seed, eval(params['GNN_ARCH'])
    if module == 'gnn':
        return module, note, seed, eval(params['MODELS'])[0]
    return module, note, seed, ''


def completed_identities(prefix: str) -> set:
    """Identities of all runs with a completed (status 'done') archive of the given prefix."""
    done = set()
    for meta_path in glob.glob(os.path.join(PATH, 'results', 'predict_molecules__*', '*', 'experiment_meta.json')):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            continue
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
            continue
        module = os.path.basename(os.path.dirname(os.path.dirname(meta_path))).replace('predict_molecules__', '')
        arch = params.get('GNN_ARCH') if module == 'gnn_random' else (params['MODELS'][0] if module == 'gnn' else '')
        done.add((module, params['NOTE'], params['SEED'], arch))
    return done


def write_missing(prefix: str, stem: str = ''):
    """
    Write the command lines without a completed archive to missing_hdf.txt / missing_gnn.txt.

    The HDF encoding cache of a (DATASET_NAME, SEED) is written only by its primer run. If that run did
    not complete, its cache file may be truncated (pycomex compresses in place, without a temp file), and
    a truncated cache would make every later run of that key fail. Such cache files are deleted here so
    that the re-run primer rebuilds them. Only call this while no ex_14 job is running.
    """
    done = completed_identities(prefix)
    hdf = open(os.path.join(OUT, f'{stem}hdf.txt')).read().splitlines()
    primer_count = int(open(os.path.join(OUT, f'{stem}hdf_primer_count.txt')).read())
    gnn = open(os.path.join(OUT, f'{stem}gnn.txt')).read().splitlines()

    missing_primer = [l for l in hdf[:primer_count] if identity_of_line(l) not in done]
    missing_rest = [l for l in hdf[primer_count:] if identity_of_line(l) not in done]
    missing_gnn = [l for l in gnn if identity_of_line(l) not in done]

    for line in missing_primer:
        params = dict(re.findall(r'--(\w+)="([^"]*)"', line))
        key = f"hdc_{eval(params['DATASET_NAME'])}__seed_{eval(params['SEED'])}__size_{eval(params['EMBEDDING_SIZE'])}__depth_{eval(params['NUM_LAYERS'])}"
        for path in glob.glob(os.path.join(PATH, '.cache', key + '.pkl*')):
            print(f'removing possibly incomplete cache {os.path.basename(path)}')
            os.remove(path)

    _write(f'missing_{stem}hdf.txt', missing_primer + missing_rest)
    _write(f'missing_{stem}hdf_primer_count.txt', [str(len(missing_primer))])
    _write(f'missing_{stem}gnn.txt', missing_gnn)


if __name__ == '__main__':
    if 'missing' in sys.argv[1:]:
        if 'smoke' in sys.argv[1:]:
            write_missing('ex_14_smoke', stem='smoke_')
        else:
            write_missing(PREFIX)
        sys.exit(0)
    smoke = 'smoke' in sys.argv[1:]
    if smoke:
        # The exact production pipeline on FreeSolv, seed 0, but a short training budget so that the
        # whole thing finishes in minutes. Validates imports, parameters, GPU use and the archive layout.
        datasets = [d for d in DATASETS if d[0] == 'freesolv_hfe']
        overrides = {'trained': {'EPOCHS': 30, 'EARLY_STOPPING_PATIENCE': 5}}
        hdf_primer, hdf_rest, gnn = build(datasets, [0], 'ex_14_smoke', overrides)
        _write('smoke_warmup.txt', build_warmup(datasets))
        _write('smoke_hdf.txt', hdf_primer + hdf_rest)
        _write('smoke_hdf_primer_count.txt', [str(len(hdf_primer))])
        _write('smoke_gnn.txt', gnn)
    else:
        hdf_primer, hdf_rest, gnn = build(DATASETS, SEEDS, PREFIX)
        _write('warmup.txt', build_warmup(DATASETS))
        _write('hdf.txt', hdf_primer + hdf_rest)
        _write('hdf_primer_count.txt', [str(len(hdf_primer))])
        _write('gnn.txt', gnn)
