"""
Command-file generator for Experiment 13 (ex_13): the HPO-first main comparison table.

On JUPITER a SLURM job gets a whole Booster node (288 cores + 4 GPUs), so submitting one
small job per experiment is wasteful. Instead this script writes plain-text command files
(one ``python predict_molecules__*.py ...`` invocation per line) that ``run_ex13.sbatch``
feeds to GNU ``parallel`` to pack hundreds of tasks onto a single node.

Protocol (Design A, validation-selected HPO -> frozen config -> multi-seed eval):

* ``hpo``   full grid (fingerprint size x radius, HDC dim x depth, x the 9 MLP configs) on a
            single tuning seed (420); big datasets are subsampled (NUM_DATA=0.1). Split into a
            *primer* pass (one command per unique featurization group, so the expensive HDC
            encodings and the shared ``load__``/``stats_`` caches are built exactly once, before
            their 8 MLP-siblings run) and a *rest* pass (everything else, all cache hits).
* ``table`` the frozen best config per (representation, dataset) from
            ``experiment_best_parameters_map__ex13.json`` (written by analyze_ex_13), evaluated
            over seeds 0-4 on full data. Split into a CPU file and a GPU file (the big datasets
            -- COMPAS, QM9 -- go to the 4 GH200s; everything else is CPU-packed).

Usage:
    python _slurm_ex_13.py hpo   [smoke]
    python _slurm_ex_13.py table [smoke]
"""
import os
import sys
import json
import itertools

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex13')
_REPO = os.path.abspath(os.path.join(PATH, os.pardir, os.pardir, os.pardir))
SHERLOCK_DICT = os.path.join(_REPO, 'data', 'sherlock', 'sherlock_r6_coconut_lotus_dict.pkl')

PREFIX_HPO = 'ex_13_hpo'
PREFIX_TABLE = 'ex_13_table'
HPO_SEED = 420
TABLE_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 10 eval repetitions for the reported table
HPO_BIG_NUM_DATA = 0.1          # subsample the big datasets during HPO only

# The 19 regression targets of the main table.
# (note, DATASET_NAME, TARGET_INDEX, extra_params, big)
#   note          -> grouping key used by analyze_ex_13 (must be unique)
#   DATASET_NAME  -> drives the load/featurization cache key (clogp shares aqsoldb's molecules)
#   big           -> subsample during HPO, route to GPU during the table eval
DATASETS = [
    ('aqsoldb_logs',       'aqsoldb',      None, {}, False),
    ('clogp',              'aqsoldb',      None, {'__INCLUDE__': 'mixin_clogp.py'}, False),
    ('freesolv_hfe',       'freesolv',     0,    {}, False),
    ('lipophilicity_logD', 'lipophilicity',0,    {}, False),
    ('bace_ic50',          'bace_reg',     0,    {}, False),
    ('hopv15_gap',         'hopv15_exp',   2,    {}, False),
    ('hopv15_jsc',         'hopv15_exp',   3,    {}, False),
    ('hopv15_voc',         'hopv15_exp',   4,    {}, False),
    ('hopv15_pce',         'hopv15_exp',   5,    {}, False),
    ('compas_dipole',      'compas_3x',    3,    {}, True),
    ('compas_gap',         'compas_3x',    2,    {}, True),
    ('compas_energy',      'compas_3x',    4,    {}, True),
    ('qm9_dipole',         'qm9_smiles',   3,    {}, True),
    ('qm9_alpha',          'qm9_smiles',   4,    {}, True),
    ('qm9_gap',            'qm9_smiles',   7,    {}, True),
    ('qm9_energy',         'qm9_smiles',   10,   {}, True),
    ('qm9_zpve',           'qm9_smiles',   9,    {}, True),
    ('qm9_enthalpy',       'qm9_smiles',   12,   {}, True),
    ('qm9_cv',             'qm9_smiles',   14,   {}, True),
]

# representation -> (encoding module suffix, featurization grid)
REPS = {
    'morgan':  ('fp',  {'FINGERPRINT_TYPE': ['morgan'],  'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192], 'FINGERPRINT_RADIUS': [1, 2, 3]}),
    'rdkit':   ('fp',  {'FINGERPRINT_TYPE': ['rdkit'],   'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192], 'FINGERPRINT_RADIUS': [2, 3, 4]}),
    'atom':    ('fp',  {'FINGERPRINT_TYPE': ['atom'],    'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192]}),
    'torsion': ('fp',  {'FINGERPRINT_TYPE': ['torsion'], 'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192]}),
    'hdc':     ('hdc', {'EMBEDDING_SIZE': [1024, 2048, 4096, 8192], 'NUM_LAYERS': [1, 2, 3]}),
    # --- additional baselines (ex_13 second wave), merged into the same table ---
    'count_morgan': ('fp', {'FINGERPRINT_TYPE': ['count_morgan'], 'FINGERPRINT_SIZE': [1024, 2048, 4096, 8192], 'FINGERPRINT_RADIUS': [1, 2, 3]}),
    'sherlock':     ('sherlock', {'FINGERPRINT_SIZE': [16384], 'SHERLOCK_RADIUS': [6], 'SHERLOCK_DICTIONARY_PATH': [SHERLOCK_DICT]}),
}
MLP_GRID = {
    'NN_HIDDEN_LAYER_SIZES': [(10, 10), (50, 50), (100, 100)],
    'NN_LEARNING_RATE_INIT': [0.0001, 0.001, 0.01],
}
# The one MLP config used for the primer pass (its featurization is what gets cached first).
PRIMER_MLP = {'NN_HIDDEN_LAYER_SIZES': (100, 100), 'NN_LEARNING_RATE_INIT': 0.001}


def _dict_product(d: dict):
    keys = list(d.keys())
    for vals in itertools.product(*[d[k] for k in keys]):
        yield dict(zip(keys, vals))


def _command(module: str, prefix: str, seed: int, num_data: float,
             ds: tuple, feat: dict, mlp: dict, models=('neural_net2',)) -> str:
    note, name, tidx, extra, _big = ds
    params = {
        '__DEBUG__': False, '__PREFIX__': prefix, 'SEED': seed,
        'NUM_TEST': 0.1, 'NUM_TRAIN': 1.0, 'NUM_DATA': num_data,
        'DATASET_NAME': name, 'DATASET_TYPE': 'regression', 'NOTE': note,
        'MODELS': list(models), 'NN_NUM_WORKERS': 0,
    }
    if tidx is not None:
        params['TARGET_INDEX'] = tidx
    params.update(feat)
    params.update(mlp)
    params.update(extra)
    parts = ['python', os.path.join(PATH, f'predict_molecules__{module}.py')]
    parts += [f'--{k}="{v!r}"' for k, v in params.items()]
    return ' '.join(parts)


# Cheapest config that still triggers the load__/stats_ cache builds (a tiny HDC encode + a fast
# linear fit). One per unique (DATASET_NAME, NUM_DATA) is run as a barrier before the parallel passes,
# so the dataset-level caches -- which every representation of a dataset shares -- are built exactly
# once instead of raced by hundreds of concurrent tasks.
WARMUP_HDC = {'EMBEDDING_SIZE': 1024, 'NUM_LAYERS': 1}


def build_warmup(datasets, stage: str, reps=None):
    reps = reps or {}
    seen, lines = set(), []
    for ds in datasets:
        _note, name, _tidx, _extra, big = ds
        num_data = (HPO_BIG_NUM_DATA if big else 1.0) if stage == 'hpo' else 1.0
        key = (name, num_data)
        if key in seen:
            continue
        seen.add(key)
        lines.append(_command('hdc', 'ex_13_warmup', HPO_SEED, num_data, ds, WARMUP_HDC,
                              {'NN_LEARNING_RATE_INIT': 0.001}, models=('linear',)))
        # Sherlock's transform cache is shared across a dataset's seeds/targets (keyed by DATASET_NAME,
        # not NOTE), so many table tasks would otherwise race to write it. Pre-build it once per dataset
        # here (distinct keys -> race-free) so the table tasks all hit a warm cache.
        if 'sherlock' in reps:
            lines.append(_command('sherlock', 'ex_13_warmup', HPO_SEED, num_data, ds,
                                  {'FINGERPRINT_SIZE': 16384, 'SHERLOCK_RADIUS': 6, 'SHERLOCK_DICTIONARY_PATH': SHERLOCK_DICT},
                                  {'NN_LEARNING_RATE_INIT': 0.001}, models=('linear',)))
    _write(f'warmup_{stage}.txt', lines)


def _write(name: str, lines: list):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'  wrote {len(lines):>5} commands -> {path}')


def build_hpo(datasets, reps):
    primer, rest = [], []
    for ds in datasets:
        big = ds[4]
        num_data = HPO_BIG_NUM_DATA if big else 1.0
        for rep, (module, fgrid) in reps.items():
            for feat in _dict_product(fgrid):
                primer.append(_command(module, PREFIX_HPO, HPO_SEED, num_data, ds, feat, PRIMER_MLP))
                for mlp in _dict_product(MLP_GRID):
                    if mlp == PRIMER_MLP:
                        continue
                    rest.append(_command(module, PREFIX_HPO, HPO_SEED, num_data, ds, feat, mlp))
    _write('hpo_primer.txt', primer)
    _write('hpo_rest.txt', rest)
    print(f'HPO total: {len(primer) + len(rest)} commands ({len(primer)} groups x 9 MLP configs)')


def build_table(datasets, reps):
    json_path = os.path.join(PATH, 'experiment_best_parameters_map__ex13.json')
    if not os.path.exists(json_path):
        sys.exit(f'ERROR: {json_path} not found -- run the HPO stage and analyze_ex_13 selection first.')
    best = {tuple(k): v for k, v in json.load(open(json_path))}  # {(rep, note): {param: value}}
    cpu, gpu = [], []
    missing = []
    for ds in datasets:
        note, name, tidx, extra, big = ds
        for rep, (module, _fgrid) in reps.items():
            key = (rep, note)
            if key not in best:
                missing.append(key)
                continue
            cfg = best[key]
            feat = {k: v for k, v in cfg.items() if k in ('FINGERPRINT_TYPE', 'FINGERPRINT_SIZE', 'FINGERPRINT_RADIUS', 'EMBEDDING_SIZE', 'NUM_LAYERS', 'SHERLOCK_RADIUS', 'SHERLOCK_DICTIONARY_PATH')}
            mlp = {k: v for k, v in cfg.items() if k in ('NN_HIDDEN_LAYER_SIZES', 'NN_LEARNING_RATE_INIT')}
            for seed in TABLE_SEEDS:
                cmd = _command(module, PREFIX_TABLE, seed, 1.0, ds, feat, mlp)
                # Big HDC jobs are RAM-heavy on CPU (full-size embeddings for 40k-134k molecules) but
                # fit comfortably in 96 GB of HBM -> route them to the 4 GH200s. Big FP jobs are ~6 GB
                # each and 4x more numerous, so they pack better across the CPU cores than behind 4 GPUs.
                (gpu if (big and rep in ('hdc', 'sherlock')) else cpu).append(cmd)
    if missing:
        print(f'WARNING: {len(missing)} (rep,dataset) cells missing from the selection JSON: {missing[:8]}')
    _write('table_cpu.txt', cpu)
    _write('table_gpu.txt', gpu)
    print(f'TABLE total: {len(cpu) + len(gpu)} commands ({len(cpu)} cpu, {len(gpu)} gpu)')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'hpo'
    smoke = 'smoke' in sys.argv[2:]
    datasets = DATASETS
    reps = REPS
    if smoke:
        # Tiny slice to validate the whole pipeline end-to-end.
        datasets = [d for d in DATASETS if d[0] in ('freesolv_hfe', 'aqsoldb_logs')]
        reps = {
            'morgan': ('fp',  {'FINGERPRINT_TYPE': ['morgan'], 'FINGERPRINT_SIZE': [1024], 'FINGERPRINT_RADIUS': [2]}),
            'hdc':    ('hdc', {'EMBEDDING_SIZE': [1024], 'NUM_LAYERS': [2]}),
        }
        print('[SMOKE] freesolv+aqsoldb, morgan(1024,r2)+hdc(1024,d2) only')
    reps_arg = next((a for a in sys.argv[2:] if a.startswith('reps=')), None)
    if reps_arg and not smoke:
        reps = {k: REPS[k] for k in reps_arg.split('=', 1)[1].split(',') if k in REPS}
        print(f'[reps filter] {list(reps)}')
    if stage == 'hpo':
        build_warmup(datasets, 'hpo', reps)
        build_hpo(datasets, reps)
    elif stage == 'table':
        build_warmup(datasets, 'table', reps)
        build_table(datasets, reps)
    else:
        sys.exit('usage: python _slurm_ex_13.py {hpo|table} [smoke]')
