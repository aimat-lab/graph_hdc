"""
Command-file generator for Experiment 14 (ex_14): comparison of HDF with graph neural networks.

Answers reviewer comment R2.1 ("missing GNN baseline"): do the gains of hyperdimensional fingerprints come
from the HDC algebra or simply from multi-hop message passing? Two standard PyG GNNs (GIN, GATv2) are each
evaluated in two ways and compared against HDF with the same downstream models:

* ``trained``  the GNN trained end-to-end on the target (fixed standard config: 3x128 conv, dense head
               128-64-32, lr 1e-4 with cosine decay to 1e-6 over the epochs, batch 32) for the full 1000
               epochs (no early stopping), after which the weights of the epoch with the best validation
               metric are restored.
* ``random``   the same architecture with frozen random weights (width 2048 = HDF dimension, 2 layers =
               HDF depth, sum pooling) as a training-free encoder, followed by the MLP and KNN.
* ``hdf``      HDF (D=2048, L=2; the ex_13 fixed config) followed by the same MLP and KNN.

All GNNs receive exactly the atom information HDF encodes (NODE_FEATURES='hdf'). The MLP is the ex_13
fixed-table MLP ((100, 100), lr 1e-3); KNN uses k=5 with Euclidean distance for every representation.

Rounds under the same prefix (only archives with the CURRENT settings count, here and in analyze_ex_14.py):

1. trained GNNs with early stopping (patience 50), which stopped the extensive QM9 targets after ~50 epochs
   on the noisy validation plateau of the fixed learning rate; replaced by full-length runs.
2. up to 2026-10-07: one-directional HDF message passing, implicit hydrogen counts (for HDF and for the GNN
   inputs: 0 for every bracket atom such as [nH] or [NH3+]), constant learning rate, and GCN as a third
   architecture.
3. 2026-10-08: all variants re-run with bidirectional HDF message passing, total hydrogen counts everywhere and
   the cosine learning-rate decay; GCN dropped (the SI reports GIN and GATv2 only). The splits are unchanged:
   they come from the cached, seed-independent dataset order (``load__`` caches).
4. 2026-10-09 (current): only the HDF arm re-run, with the unit-modulus codebooks that became the HDF default
   (SPECTRUM='unit', ex_22; size/diameter encodings kept); the GNN runs of round 3 remain current. Same splits.
   The setting is added to the HDF commands of ex_14 only (HDF_ROUND), not to VARIANTS, which other schedulers
   import; their HDF commands keep the original codebooks (see _command).

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
* ``gnn_trained.txt``  only the trained-GNN lines of gnn.txt (same cost ordering), for re-running them.
* ``smoke_*``     the same pipeline on FreeSolv with seed 0, all 5 variants, few epochs.

* ``missing_*``   written by the ``missing`` mode: the lines of hdf.txt / gnn.txt that have no completed
                  (status 'done') archive with the CURRENT settings yet, e.g. after a timeout, a node failure
                  or failed runs. Submit them with HDF_FILE / GNN_FILE pointing to these files (see
                  run_ex14_kcist.sbatch).

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

ARCHS = ['gin', 'gatv2']

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
    # module suffix, params (MODELS for the trained GNN is filled in per architecture). The HDF encoder
    # settings (BIDIRECTIONAL, HYDROGEN_COUNT) are added by _command.
    'hdf': ('hdc', {
        'EMBEDDING_SIZE': 2048, 'NUM_LAYERS': 2, 'ENCODING_MODE': 'continuous', 'DEVICE': 'cpu',
        **DOWNSTREAM,
    }),
    'random': ('gnn_random', {
        'EMBEDDING_SIZE': 2048, 'NUM_LAYERS': 2, 'NODE_FEATURES': 'hdf', 'HYDROGEN_COUNT': 'total',
        'DEVICE': 'cuda',
        **DOWNSTREAM,
    }),
    'trained': ('gnn', {
        'NODE_FEATURES': 'hdf', 'HYDROGEN_COUNT': 'total', 'CONV_UNITS': [128, 128, 128],
        'DENSE_UNITS': [128, 64, 32], 'BATCH_SIZE': 32, 'LEARNING_RATE': 1e-4, 'LR_SCHEDULE': 'cosine',
        'LR_MIN': 1e-6, 'EPOCHS': 1000, 'EARLY_STOPPING_PATIENCE': None,
    }),
}

# The settings that distinguish the current round from the earlier ones under the same prefix (see the
# module docstring), per module suffix. Only archives with all of them count as done (missing mode) and
# enter the analysis (analyze_ex_14.py); build() checks that every generated command has them.
CURRENT = {
    'hdc': {'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'SPECTRUM': 'unit'},
    'gnn_random': {'HYDROGEN_COUNT': 'total'},
    'gnn': {'HYDROGEN_COUNT': 'total', 'LR_SCHEDULE': 'cosine', 'EARLY_STOPPING_PATIENCE': None},
}


# HDF settings of the current round that are added to the HDF commands of ex_14 only: VARIANTS is imported by other
# schedulers (ex_15, ex_18, ex_20, ex_22), whose HDF commands keep the original codebooks.
HDF_ROUND = {'SPECTRUM': 'unit'}


def is_current(module: str, params: dict) -> bool:
    """Whether a run of ``predict_molecules__<module>`` with these parameters belongs to the current round."""
    return all(params.get(key) == value for key, value in CURRENT.get(module, {}).items())


def params_of_line(line: str) -> dict:
    """The parameters of a command line, as Python values."""
    return {k: eval(v) for k, v in re.findall(r'--(\w+)="([^"]*)"', line)}


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
    # HDF runs use the fixed encoder: bidirectional message passing and total hydrogen counts. The archives up to
    # 2026-10-07 were produced with one-directional message passing and implicit hydrogen counts (the behavior
    # before both parameters existed); callers can pass the old values explicitly (ex_15 passes both).
    if module == 'hdc':
        full.setdefault('BIDIRECTIONAL', True)
        full.setdefault('HYDROGEN_COUNT', 'total')
        # Unit-modulus codebooks became the HDF default on 2026-10-09 (ex_22). Every HDF archive of the experiments
        # built with this function (ex_14, ex_15, ex_18, ex_20) used the original (gaussian) codebooks, so their commands
        # pin them: resumed or missing runs match their archives. Switching an experiment to "unit" is a new round
        # (add SPECTRUM to CURRENT and the "__unitspec" suffix to hdf_cache_name). ex_22 passes SPECTRUM itself.
        full.setdefault('SPECTRUM', 'gaussian')
    # Relative script path: the command files are executed from this folder (see run_ex14_kcist.sbatch),
    # so the same files work locally and on the cluster.
    parts = ['python', f'predict_molecules__{module}.py']
    parts += [f'--{k}="{v!r}"' for k, v in full.items()]
    return ' '.join(parts)


def variant_commands(ds: tuple, seed: int, prefix: str, overrides: dict = {}) -> dict:
    """All commands for one (dataset, seed), keyed by variant name ('hdf', 'random_gcn', 'trained_gin', ...)."""
    cmds = {}
    module, params = VARIANTS['hdf']
    cmds['hdf'] = _command(module, prefix, seed, ds, {**params, **HDF_ROUND, **overrides.get('hdf', {})})
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
            for line in cmds.values():
                # the missing mode and the analysis count only archives of the current round
                assert is_current(identity_of_line(line)[0], params_of_line(line)), f'not CURRENT: {line}'
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
    params = params_of_line(line)
    note, seed = params['NOTE'], params['SEED']
    if module == 'gnn_random':
        return module, note, seed, params['GNN_ARCH']
    if module == 'gnn':
        return module, note, seed, params['MODELS'][0]
    return module, note, seed, ''


def completed_identities(prefix: str) -> set:
    """Identities of all runs of the current round (CURRENT) with a completed (status 'done') archive."""
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
        if not is_current(module, params):
            continue
        arch = params.get('GNN_ARCH') if module == 'gnn_random' else (params['MODELS'][0] if module == 'gnn' else '')
        done.add((module, params['NOTE'], params['SEED'], arch))
    return done


def hdf_cache_name(params: dict) -> str:
    """Name of the HDF encoding cache of a full-data predict_molecules__hdc run (as built in that module)."""
    name = (f"hdc_{params['DATASET_NAME']}__seed_{params['SEED']}__size_{params['EMBEDDING_SIZE']}"
            f"__depth_{params['NUM_LAYERS']}")
    if params.get('BIDIRECTIONAL'):
        name += '__bidir'
    if params.get('HYDROGEN_COUNT') == 'total':
        name += '__totalh'
    if params.get('ENCODING_MODE') == 'continuous' and params.get('SPECTRUM') == 'unit':
        name += '__unitspec'
    if params.get('GRAPH_ATTRIBUTES') is False:
        name += '__noglobal'
    return name


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
        key = hdf_cache_name(params_of_line(line))
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
        overrides = {'trained': {'EPOCHS': 5}}
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
        _write('gnn_trained.txt', [line for line in gnn if 'predict_molecules__gnn.py' in line])
