"""
Command-file generator for Experiment 27 (ex_27): the re-run of Table 1 (MLP) and of the KNN table of the SI with the
current HDF encoder, on JUPITER.

Since the tables were computed, the HDF encoder changed (bidirectional message passing, total hydrogen counts,
unit-modulus codebooks) and the HOPV15 and ClogP targets were corrected, so both tables are re-run (user decision
2026-10-13), in this order:

1. ``fixed``: every method with the same standard configuration (2048 dimensions or bits, radius or depth 2) and the
   same MLP ((100, 100), learning rate 1e-3, 200 epochs, batch size 64), and the KNN table with the same KNN for every
   method (k = 5, uniform weights; the KNN is never optimized). Every run trains the MLP and then the KNN on the same
   featurization and split.
2. ``opt`` / ``opteval``: every method gets the same small Bayesian-optimization budget, 30 trials of TPE per method and
   target (_optuna_ex_27.py), on a tuning split; the best configuration is then evaluated over the 10 seeds (MLP only).

Common settings: the 17 targets of Table 1, the five columns HDF, Morgan, RDKit, topological torsion and atom pair,
the seeds 0-9 (split and network), and one fixed HDF codebook for all seeds (CODEBOOK_SEED 0), so that, as for the
fingerprints, the features of a molecule are the same in every repetition. The KNN uses the native distance of each
representation: the Jaccard (Tanimoto) distance for the bit vectors and the cosine distance for HDF.

Efficiency (user decision 2026-10-13): the MLPs train on the GPUs with the data held on the device (NN_DEVICE_DATA),
the KNN searches on the GPU (KN_BACKEND 'torch': exact, ties broken deterministically), and the targets of a dataset
share one run (TARGET_INDICES: one featurization and split, then one MLP and one KNN per target). On the CPU, these
options reproduce the earlier code exactly (NN_DEVICE_DATA, TARGET_INDICES) or up to the choice among tied neighbors
(KNN). The stage ``check`` compares the GPU with the CPU before the production runs.

Stages, with their files in ``_ex27/`` (``run_ex27.sbatch`` runs a stage on one Booster node: the warmup file first,
as a barrier, then the GPU file and the CPU file side by side):

* ``check`` (prefix ex_27_check): FreeSolv, AqSolDB and the QM9 gap; HDF and Morgan; seeds 0-2; four variants:
    ``cpu``         the data path of the earlier tables (NN_DEVICE_DATA off) on the CPU (the runner hides the GPUs);
                    KNN with the torch backend on the CPU
    ``cpu_prod``    (prefix ex_27_cpuprod) the production code path on the CPU: NN_DEVICE_DATA, and for QM9 the gap as
                    the one target of a TARGET_INDICES run. It must reproduce the cpu variant exactly, so that the
                    differences between gpu and cpu come from the GPU arithmetic alone.
    ``gpu``         the production configuration on a GPU; QM9 with all seven targets in one run (TARGET_INDICES),
                    which also gives the run time and the GPU memory of a production QM9 run
    ``cpu_nnseed``  the cpu variant with another network seed (NN_SEED = SEED + 100), MLP only: the run-to-run
                    variation of the training, against which the differences between cpu and gpu are judged
  The same job runs the smoke test of the fixed stage (prefix ex_27_smoke): seed 0 of the production commands of the
  small groups HOPV15 (two targets in one run), FreeSolv and ClogP (mixin_clogp), all five methods, on the GPUs.
  Files: check_warmup.txt, check_gpu.txt (gpu variant + smoke), check_cpu.txt (the three CPU variants)
* ``fixed`` (prefix ex_27_fixed): 8 dataset groups x 5 methods x 10 seeds = 400 runs, all on the GPUs.
  Files: fixed_warmup.txt, fixed_gpu.txt (longest first)
* ``opt``: one Optuna study per method and target (5 x 17 = 85 studies of OPT_TRIALS trials), each a process of
  _optuna_ex_27.py that runs its trials one after the other on the GPU of its slot. A trial is a run (prefix
  <PREFIX_TRIAL>__<method>__<target>__<number>) with the trial's parameters on the tuning split: SEED OPT_SEED (as the
  tuning seed of ex_13, not one of the evaluation seeds), the full dataset, MLP only; the objective is its validation
  MAE. Search spaces: see suggest_params (user decisions 2026-10-13). Round 1 tuned QM9 and COMPAS-3X on a 10%
  subsample; round 2 re-tunes these 50 studies on the full data (see OPT_DIR; stage optfull_a / optfull_b).
  Files: opt_primer.txt and opt_warmup.txt (every HDF encoding the trials can ask for), opt_gpu.txt (the studies)
* ``opteval`` (prefix ex_27_opt): the best configuration of every study, evaluated over the seeds 0-9 on the full data
  (MLP only; one run per target, as the configurations differ). Written by ``python _slurm_ex_27.py opteval`` after the
  studies: opteval_primer.txt and opteval_warmup.txt (the HDF encodings of the chosen configurations), opteval_gpu.txt;
  ``python _slurm_ex_27.py opteval_missing`` writes opteval_missing.txt (the runs without a completed archive)

The warmups encode the datasets once with the fixed codebook (short runs without models), so that the parallel runs
only read the HDF encoding caches, which none of them could write without racing the others. The runs of a dataset
(subsample) also share its dataset caches (load__, stats_): the first warmup of every dataset forms the primer, which
the runner completes before the other warmups. The full-data dataset caches also fix the molecule order, and thereby
the splits of every stage: they are written once (on JUPITER they were copied from the ex_13 checkout) and must not be
rebuilt while runs are going. The runner stops if a primer or warmup run fails, because pycomex writes its caches in
place and a failed run can leave a truncated cache file behind; delete that file before the next attempt.

Usage (from the fingerprints folder of the ex_27 worktree on JUPITER):
    python _slurm_ex_27.py              # writes all command files
    sbatch --export=ALL,EX27_STAGE=check --time=04:00:00 run_ex27.sbatch
    python analyze_ex_27.py check
    sbatch --export=ALL,EX27_STAGE=fixed run_ex27.sbatch
    python _slurm_ex_27.py missing      # after an interrupted fixed stage (no ex_27 job running): fixed_missing.txt
    sbatch --export=ALL,EX27_STAGE=fixed,GPU_FILE=fixed_missing.txt run_ex27.sbatch
    python _slurm_ex_27.py fixed_split  # alternatively: the (remaining) fixed stage as two shorter jobs side by side
    sbatch --export=ALL,EX27_STAGE=fixed_a --time=04:00:00 run_ex27.sbatch     (and the same with fixed_b)
    sbatch --export=ALL,EX27_STAGE=opt,GPU_J=88 run_ex27.sbatch     # an interrupted study resumes when resubmitted
    python _slurm_ex_27.py opt_split    # alternatively: two shorter jobs side by side (stages opt_a and opt_b)
    python _slurm_ex_27.py opt_full_init   # round 2, once: the 35 full-data journals of round 1 into OPT_DIR
    python _slurm_ex_27.py optfull_split   # round 2: stages optfull_a (warmup + HDF studies) and optfull_b; a study
                                           # needs more than one 4 h job, so resubmit (or chain) until all are done
    python _slurm_ex_27.py opteval      # after all studies are complete
    sbatch --export=ALL,EX27_STAGE=opteval run_ex27.sbatch
    python _slurm_ex_27.py opteval_split   # alternatively: stages opteval_a and opteval_b
    python _slurm_ex_27.py opteval_missing   # after an interrupted opteval (no ex_27 job running)
    sbatch --export=ALL,EX27_STAGE=opteval,GPU_FILE=opteval_missing.txt run_ex27.sbatch
    python _slurm_ex_27.py opteval_missing_split   # the same as two jobs (stages opteval_missing_a / _b)
    (opteval with up to 8192 dimensions: a QM9 run needs up to ~8 GB of GPU memory, so GPU_J=24, not 48)
    python analyze_ex_27.py opt
"""
import os
import re
import sys
import glob
import json

from _slurm_ex_13 import DATASETS as EX13_DATASETS
from _slurm_ex_14 import params_of_line
from _slurm_ex_25 import TABLE1_NOTES

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex27')

PREFIX_WARMUP = 'ex_27_warmup'
PREFIX_CHECK = 'ex_27_check'
PREFIX_CPUPROD = 'ex_27_cpuprod'
PREFIX_SMOKE = 'ex_27_smoke'
PREFIX_FIXED = 'ex_27_fixed'
SEEDS = list(range(10))
CHECK_SEEDS = [0, 1, 2]
CODEBOOK_SEED = 0

# The targets of a dataset share one run: (group, DATASET_NAME, the notes of its Table 1 targets, extra parameters).
# The notes, TARGET_INDEX values and extra parameters are those of _slurm_ex_13 (with the corrected HOPV15 columns).
GROUPS = [
    ('qm9', 'qm9_smiles', ['qm9_energy', 'qm9_enthalpy', 'qm9_zpve', 'qm9_alpha', 'qm9_cv', 'qm9_gap', 'qm9_dipole'], {}),
    ('compas', 'compas_3x', ['compas_energy', 'compas_gap', 'compas_dipole'], {}),
    ('hopv15', 'hopv15_exp', ['hopv15_pce', 'hopv15_voc'], {}),
    ('aqsoldb', 'aqsoldb', ['aqsoldb_logs'], {}),
    # mixin_clogp replaces the labels in a hook, which TARGET_INDICES does not allow: ClogP has its own runs
    ('clogp', 'aqsoldb', ['clogp'], {'__INCLUDE__': 'mixin_clogp.py'}),
    ('freesolv', 'freesolv', ['freesolv_hfe'], {}),
    ('bace', 'bace_reg', ['bace_ic50'], {}),
    ('lipophilicity', 'lipophilicity', ['lipophilicity_logD'], {}),
]
GROUP = {group: (name, notes, extra) for group, name, notes, extra in GROUPS}
TARGET_INDEX = {note: tidx for note, _name, tidx, _extra, _big in EX13_DATASETS}
assert sorted(note for *_, notes, _ in GROUPS for note in notes) == sorted(TABLE1_NOTES)
assert all((name, extra) == next((n, x) for note_, n, _, x, _ in EX13_DATASETS if note_ == note)
           for _, name, notes, extra in GROUPS for note in notes)

# The HDF standard encoder (all values are the current defaults of predict_molecules__hdc; spelled out so that the
# commands do not depend on them) with one codebook for all seeds. The encoding runs on the CPU (in the warmup).
HDF = {
    'EMBEDDING_SIZE': 2048, 'NUM_LAYERS': 2, 'ENCODING_MODE': 'continuous', 'SPECTRUM': 'unit', 'BIDIRECTIONAL': True,
    'HYDROGEN_COUNT': 'total', 'GRAPH_ATTRIBUTES': True, 'CODEBOOK_SEED': CODEBOOK_SEED, 'DEVICE': 'cpu',
}
# (method, module suffix, featurization, KNN metric). RDKit: paths of up to 2 x radius = 4 bonds; torsion and atom
# pair with the RDKit defaults (4 atoms, distances up to 30), as in the fixed table of ex_13.
METHODS = [
    ('hdf', 'hdc', HDF, 'cosine'),
    ('morgan', 'fp', {'FINGERPRINT_TYPE': 'morgan', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 2}, 'jaccard'),
    ('rdkit', 'fp', {'FINGERPRINT_TYPE': 'rdkit', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 2}, 'jaccard'),
    ('torsion', 'fp', {'FINGERPRINT_TYPE': 'torsion', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_TORSION_ATOMS': 4},
     'jaccard'),
    ('atompair', 'fp', {'FINGERPRINT_TYPE': 'atom', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_MAX_DISTANCE': 30},
     'jaccard'),
]
METHOD = {method: (module, feat, metric) for method, module, feat, metric in METHODS}
FINGERPRINT_METHOD = {feat['FINGERPRINT_TYPE']: method for method, module, feat, _ in METHODS if module == 'fp'}

MLP = {'NN_HIDDEN_LAYER_SIZES': (100, 100), 'NN_LEARNING_RATE_INIT': 0.001}
KNN = {'KN_NUM_NEIGHBORS': 5, 'KN_WEIGHTS': 'uniform', 'KN_BACKEND': 'torch'}
# the production configuration: MLP and KNN in one run, the MLP data on the device
PRODUCTION = {'MODELS': ['neural_net2', 'k_neighbors'], **MLP, **KNN, 'NN_DEVICE_DATA': True}

CHECK_TARGETS = ['freesolv_hfe', 'aqsoldb_logs', 'qm9_gap']
CHECK_METHODS = ['hdf', 'morgan']
SMOKE_GROUPS = ['hopv15', 'freesolv', 'clogp']
NN_SEED_OFFSET = 100

# --- optimized table (stages opt and opteval) ---
# Round 2 (user decision 2026-10-16): every study tunes on the full dataset. Round 1 (journals in _ex27/optuna, trials
# ex_27_trial__*) tuned QM9 and COMPAS-3X on a 10% subsample, as ex_13 did; with the epochs fixed, its trials made 10x
# fewer gradient steps, and the configurations chosen there (larger learning rates, smaller networks and embeddings)
# transferred poorly to the full data. The 35 studies of the other datasets were already tuned on the full data and are
# taken over unchanged (opt_full_init copies their journals); the 50 studies of ROUND2_GROUPS are run again.
PREFIX_TRIAL = 'ex_27_trialfull'
PREFIX_OPT = 'ex_27_opt'
OPT_TRIALS = 30
OPT_SEED = 420
# the groups whose datasets are subsampled on the tuning split (round 1: {'qm9': 0.1, 'compas': 0.1})
OPT_SUBSAMPLED = {}
OPT_DIR = os.path.join(OUT, 'optuna_full')
OPT_DIR_ROUND1 = os.path.join(OUT, 'optuna')
ROUND2_GROUPS = ('qm9', 'compas')
SIZES = [1024, 2048, 4096, 8192]
HDF_DEPTHS = [1, 2, 3]
HIDDEN = {'10,10': (10, 10), '50,50': (50, 50), '100,100': (100, 100)}


def suggest_params(trial, method: str) -> dict:
    """
    The parameters of one Optuna trial of a method (user decisions 2026-10-13): the size (1024-8192) and one
    structural parameter of every representation, for HDF its depth and whether the graph size and diameter are
    encoded, and the same MLP space for all (hidden layers as in ex_13, learning rate log-uniform in [1e-4, 1e-2]).
    """
    module, feat, _ = METHOD[method]
    params = dict(feat)
    if method == 'hdf':
        params['EMBEDDING_SIZE'] = trial.suggest_categorical('size', SIZES)
        params['NUM_LAYERS'] = trial.suggest_categorical('depth', HDF_DEPTHS)
        params['GRAPH_ATTRIBUTES'] = trial.suggest_categorical('global_descriptors', [True, False])
    else:
        params['FINGERPRINT_SIZE'] = trial.suggest_categorical('size', SIZES)
        if method == 'morgan':
            params['FINGERPRINT_RADIUS'] = trial.suggest_categorical('radius', [1, 2, 3])
        elif method == 'rdkit':
            # predict_molecules__fp builds RDKit fingerprints from paths of up to 2 x FINGERPRINT_RADIUS bonds
            params['FINGERPRINT_RADIUS'] = trial.suggest_categorical('max_path_half', [1, 2, 3, 4])
        elif method == 'atompair':
            params['FINGERPRINT_MAX_DISTANCE'] = trial.suggest_categorical('max_distance', [4, 8, 16, 30])
        elif method == 'torsion':
            params['FINGERPRINT_TORSION_ATOMS'] = trial.suggest_categorical('torsion_atoms', [3, 4, 5, 6])
    params['NN_HIDDEN_LAYER_SIZES'] = HIDDEN[trial.suggest_categorical('hidden', list(HIDDEN))]
    params['NN_LEARNING_RATE_INIT'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    return params


def group_of_note(note: str) -> str:
    return next(group for group, _, notes, _ in GROUPS if note in notes)


def _command(module: str, prefix: str, seed: int, group: str, notes: list, params: dict, multi: bool = None) -> str:
    """
    One ``python predict_molecules__<module>.py --K="repr(v)" ...`` line (the format of ex_13 / ex_14) for the given
    targets of a group: a single target as in the earlier experiments (NOTE = its note, TARGET_INDEX), several as one
    run with TARGET_INDICES (NOTE = the group). ``multi`` forces the TARGET_INDICES form for a single target.
    """
    name, group_notes, extra = GROUP[group]
    assert notes and all(note in group_notes for note in notes)
    full = {
        '__DEBUG__': False, '__PREFIX__': prefix, 'SEED': seed,
        'NUM_TEST': 0.1, 'NUM_TRAIN': 1.0, 'NUM_DATA': 1.0,
        'DATASET_NAME': name, 'DATASET_TYPE': 'regression', 'NN_NUM_WORKERS': 0,
    }
    if not (len(notes) > 1 if multi is None else multi):
        assert len(notes) == 1
        full['NOTE'] = notes[0]
        if TARGET_INDEX[notes[0]] is not None:
            full['TARGET_INDEX'] = TARGET_INDEX[notes[0]]
    else:
        full['NOTE'] = group
        full['TARGET_INDICES'] = [TARGET_INDEX[note] for note in notes]
    full.update(params)
    full.update(extra)
    # relative script path: the runner executes the files from the fingerprints folder
    parts = ['python', f'predict_molecules__{module}.py']
    parts += [f'--{k}="{v!r}"' for k, v in full.items()]
    return ' '.join(parts)


def production_command(prefix: str, seed: int, group: str, method: str) -> str:
    module, feat, metric = METHOD[method]
    return _command(module, prefix, seed, group, GROUP[group][1], {**feat, **PRODUCTION, 'KN_METRIC': metric})


def warmup_commands(groups: list) -> list:
    """One HDF run per DATASET_NAME of the groups (linear model only), which writes the encoding cache."""
    lines, seen = [], set()
    for group in groups:
        name, notes, _extra = GROUP[group]
        if name in seen:
            continue
        seen.add(name)
        # the plain dataset (without the ClogP mixin) of the group's first target
        plain = next(g for g, n, _, x in GROUPS if n == name and not x)
        lines.append(_command('hdc', PREFIX_WARMUP, 0, plain, GROUP[plain][1][:1],
                              {**HDF, 'MODELS': ['linear']}))
    return lines


def plain_group(name: str) -> str:
    """The group of a DATASET_NAME without extra parameters (ClogP shares the molecules of AqSolDB)."""
    return next(g for g, n, _, x in GROUPS if n == name and not x)


def hdf_warmup_command(name: str, seed: int, num_data: float, encoder: dict) -> str:
    """A run without models (MODELS=[]) that writes the HDF encoding cache of an encoder configuration."""
    group = plain_group(name)
    return _command('hdc', PREFIX_WARMUP, seed, group, GROUP[group][1][:1],
                    {**HDF, **encoder, 'MODELS': [], 'NUM_DATA': num_data})


def opt_num_data(group: str) -> float:
    return OPT_SUBSAMPLED.get(group, 1.0)


def opt_trial_command(prefix: str, method: str, note: str, params: dict) -> str:
    """A trial: the trial's parameters (suggest_params) on the tuning split, MLP only."""
    group = group_of_note(note)
    return _command(METHOD[method][0], prefix, OPT_SEED, group, [note],
                    {**params, 'MODELS': ['neural_net2'], 'NN_DEVICE_DATA': True, 'NUM_DATA': opt_num_data(group)})


def split_primer(lines: list) -> tuple:
    """
    (primer, rest) of warmup lines: the first line of every (DATASET_NAME, NUM_DATA, SEED) goes into the primer, which
    the runner completes before the rest, because the runs of a subsample share its dataset caches (load__, stats_),
    which a run writes when they do not exist yet (e.g. the 10% tuning subsamples of QM9 and COMPAS-3X).
    """
    primer, rest, seen = [], [], set()
    for line in lines:
        params = params_of_line(line)
        key = (params['DATASET_NAME'], params['NUM_DATA'], params['SEED'])
        (rest if key in seen else primer).append(line)
        seen.add(key)
    return primer, rest


def opt_warmup_commands(groups: tuple = None) -> tuple:
    """
    (primer, rest): every HDF encoding the trials can ask for (size x depth x graph size and diameter) on the tuning
    splits of the datasets of the groups (default: all), the first of every dataset in the primer (see split_primer).
    """
    lines, seen = [], set()
    for group, name, _notes, _extra in GROUPS:
        if name in seen or (groups is not None and group not in groups):
            continue
        seen.add(name)
        for size in reversed(SIZES):
            for depth in HDF_DEPTHS:
                for attributes in (True, False):
                    encoder = {'EMBEDDING_SIZE': size, 'NUM_LAYERS': depth, 'GRAPH_ATTRIBUTES': attributes}
                    lines.append((group, hdf_warmup_command(name, OPT_SEED, opt_num_data(plain_group(name)), encoder)))
    return split_primer([line for group, line in sorted(lines, key=lambda gl: -expected_cost(gl[0]))])


def opt_full_init():
    """
    Round 2: copy the journals of the round-1 studies that were already tuned on the full data (all groups except
    ROUND2_GROUPS) into OPT_DIR, so that OPT_DIR holds all 85 studies of the final protocol. Existing files are kept.
    """
    import shutil
    os.makedirs(OPT_DIR, exist_ok=True)
    copied = 0
    for method, *_ in METHODS:
        for group, _, notes, _ in GROUPS:
            if group in ROUND2_GROUPS:
                continue
            for note in notes:
                source = os.path.join(OPT_DIR_ROUND1, f'{method}__{note}.log')
                target = os.path.join(OPT_DIR, f'{method}__{note}.log')
                if not os.path.exists(target):
                    shutil.copy2(source, target)
                    copied += 1
    print(f'copied {copied} round-1 journals of full-data studies to {os.path.relpath(OPT_DIR, PATH)}')


def optfull_split() -> tuple:
    """
    Round 2 as two jobs side by side: (primer A, warmup A, studies A, studies B) for the studies of ROUND2_GROUPS. Job A
    writes the full-data HDF encodings and runs the HDF studies after them; the fingerprint studies, which need no
    warmup, are assigned to the job with less expected work (a QM9 study counts 1, a COMPAS-3X study 0.3).
    """
    primer, warmup = opt_warmup_commands(ROUND2_GROUPS)
    studies = [(group, method, f'python _optuna_ex_27.py {method} {note}')
               for method, *_ in METHODS for group, _, notes, _ in GROUPS if group in ROUND2_GROUPS for note in notes]
    cost = {'qm9': 1.0, 'compas': 0.3}
    studies_a = [(g, line) for g, method, line in studies if method == 'hdf']
    studies_b = []
    load = {'a': sum(cost[g] for g, _ in studies_a), 'b': 0.0}
    for group, method, line in sorted(studies, key=lambda s: -cost[s[0]]):
        if method == 'hdf':
            continue
        job = min(load, key=load.get)
        (studies_a if job == 'a' else studies_b).append((group, line))
        load[job] += cost[group]
    order = lambda pairs: [line for g, line in sorted(pairs, key=lambda gl: -cost[gl[0]])]
    return primer, warmup, order(studies_a), order(studies_b)


def opt_study_commands() -> list:
    """One line per study (method x target), the longest first."""
    lines = [(group, f'python _optuna_ex_27.py {method} {note}')
             for method, *_ in METHODS for group, _, notes, _ in GROUPS for note in notes]
    return [line for group, line in sorted(lines, key=lambda gl: -expected_cost(gl[0]))]


def study_storage(method: str, note: str):
    import optuna
    path = os.path.join(OPT_DIR, f'{method}__{note}.log')
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path))


def best_trial(method: str, note: str):
    """
    The best of the first OPT_TRIALS complete trials of a study (by trial number, so that a study that got more
    complete trials, e.g. from two processes, still has the same budget as the others). Raises if the study has fewer.
    """
    import optuna
    # (loading a study whose journal does not exist would create an empty journal file)
    if not os.path.exists(os.path.join(OPT_DIR, f'{method}__{note}.log')):
        raise RuntimeError(f'study {method}__{note} does not exist')
    study = optuna.load_study(study_name=f'{method}__{note}', storage=study_storage(method, note))
    complete = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE), key=lambda t: t.number)
    if len(complete) < OPT_TRIALS:
        raise RuntimeError(f'study {method}__{note} has {len(complete)} of {OPT_TRIALS} complete trials')
    return min(complete[:OPT_TRIALS], key=lambda t: t.value)


def best_params(method: str, note: str) -> dict:
    """The run parameters of the best trial of a study (stored by _optuna_ex_27.py as the user attribute 'params')."""
    params = json.loads(best_trial(method, note).user_attrs['params'])
    params['NN_HIDDEN_LAYER_SIZES'] = tuple(params['NN_HIDDEN_LAYER_SIZES'])
    return params


def matches_params(params: dict, expected: dict) -> bool:
    """Whether the parameters of a run (as read from its archive, tuples as lists) include the expected ones."""
    def normal(value):
        return list(value) if isinstance(value, tuple) else value
    return all(normal(params.get(key)) == normal(value) for key, value in expected.items())


def opteval_commands() -> tuple:
    """
    (primer, warmup, GPU lines) of the evaluation of the best configurations over the seeds on the full data. The GPU
    lines are ordered by cost class, then seed, target and method, so that the costly HDF runs of QM9 (which load the
    largest encodings) are spread over the start instead of all starting at once.
    """
    warmup, runs, seen = [], [], set()
    for index, (method, module, *_) in enumerate(METHODS):
        for group, name, notes, _extra in GROUPS:
            for note in notes:
                params = best_params(method, note)
                if module == 'hdc':
                    encoder = {k: params[k] for k in ('EMBEDDING_SIZE', 'NUM_LAYERS', 'GRAPH_ATTRIBUTES')}
                    if (name, *encoder.values()) not in seen:
                        seen.add((name, *encoder.values()))
                        warmup.append(hdf_warmup_command(name, 0, 1.0, encoder))
                for seed in SEEDS:
                    line = _command(module, PREFIX_OPT, seed, group, [note],
                                    {**params, 'MODELS': ['neural_net2'], 'NN_DEVICE_DATA': True})
                    runs.append(((-expected_cost(group), seed, note, index), line))
    return (*split_primer(warmup), [line for key, line in sorted(runs)])


def completed_opteval() -> set:
    """(method, NOTE, SEED) of every completed opteval archive whose parameters are the best ones of its study."""
    done, best = set(), {}
    for module in ('hdc', 'fp'):
        for meta_path in glob.glob(os.path.join(PATH, 'results', f'predict_molecules__{module}', f'{PREFIX_OPT}__*',
                                                'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            if params.get('__PREFIX__') != PREFIX_OPT or meta.get('status') != 'done' or meta.get('has_error'):
                continue
            method, note = method_of_module(module, params), params.get('NOTE')
            if (method, note) not in best:
                best[(method, note)] = best_params(method, note)
            if matches_params(params, best[(method, note)]):
                done.add((method, note, params.get('SEED')))
    return done


def method_of_module(module: str, params: dict):
    """The method of a run of any configuration (opt and trial runs): HDF or the fingerprint type."""
    return 'hdf' if module == 'hdc' else FINGERPRINT_METHOD.get(params.get('FINGERPRINT_TYPE'))


def expected_cost(group: str) -> int:
    """Rough cost class of a run (higher = longer): QM9 > COMPAS-3X > the rest."""
    return {'qm9': 2, 'compas': 1}.get(group, 0)


def method_of_params(module: str, params: dict):
    """The method of a run of predict_molecules__<module> with these parameters (None if it is none of METHODS)."""
    if module == 'hdc':
        return 'hdf' if all(params.get(k) == v for k, v in HDF.items()) else None
    if module == 'fp':
        method = FINGERPRINT_METHOD.get(params.get('FINGERPRINT_TYPE'))
        if method and all(params.get(k) == v for k, v in METHOD[method][1].items()):
            return method
    return None


def check_variant(params: dict) -> str:
    """The variant of a check run: 'cpu_prod' (prefix ex_27_cpuprod), 'gpu', 'cpu' or 'cpu_nnseed'."""
    if params.get('__PREFIX__') == PREFIX_CPUPROD:
        return 'cpu_prod'
    if params.get('NN_DEVICE_DATA'):
        return 'gpu'
    return 'cpu_nnseed' if params.get('NN_SEED') is not None else 'cpu'


def check_commands() -> tuple:
    """(gpu lines, cpu lines) of the check (see the module docstring)."""
    gpu, cpu = [], []
    for seed in CHECK_SEEDS:
        for note in CHECK_TARGETS:
            group = group_of_note(note)
            for method in CHECK_METHODS:
                module, feat, metric = METHOD[method]
                cpu_params = {**feat, **MLP, **KNN, 'MODELS': ['neural_net2', 'k_neighbors'], 'KN_METRIC': metric,
                              'NN_DEVICE_DATA': False}
                production = {**feat, **PRODUCTION, 'KN_METRIC': metric}
                cpu.append(_command(module, PREFIX_CHECK, seed, group, [note], cpu_params))
                cpu.append(_command(module, PREFIX_CHECK, seed, group, [note],
                                    {**cpu_params, 'MODELS': ['neural_net2'], 'NN_SEED': seed + NN_SEED_OFFSET}))
                cpu.append(_command(module, PREFIX_CPUPROD, seed, group, [note], production, multi=group == 'qm9'))
                notes = GROUP[group][1] if group == 'qm9' else [note]
                gpu.append(_command(module, PREFIX_CHECK, seed, group, notes, production))
    # the QM9 runs first: they take longest
    gpu.sort(key=lambda line: -expected_cost(group_of_note_line(line)))
    cpu.sort(key=lambda line: -expected_cost(group_of_note_line(line)))
    return gpu, cpu


def group_of_note_line(line: str) -> str:
    note = params_of_line(line)['NOTE']
    return note if note in GROUP else group_of_note(note)


def smoke_commands() -> list:
    return [production_command(PREFIX_SMOKE, 0, group, method) for group in SMOKE_GROUPS for method, *_ in METHODS]


def fixed_commands() -> list:
    """All production runs of the fixed stage, longest first (QM9, COMPAS-3X, then the rest)."""
    lines = [(group, production_command(PREFIX_FIXED, seed, group, method))
             for seed in SEEDS for group, *_ in GROUPS for method, *_ in METHODS]
    return [line for group, line in sorted(lines, key=lambda gl: -expected_cost(gl[0]))]


def standard_hdf_cache_exists(name: str) -> bool:
    """Whether the full-data encoding of a dataset with the standard HDF encoder (HDF) is in the .cache folder."""
    key = (f'hdc_{name}__codebook_{CODEBOOK_SEED}__size_{HDF["EMBEDDING_SIZE"]}__depth_{HDF["NUM_LAYERS"]}'
           f'__bidir__totalh__unitspec')
    return bool(glob.glob(os.path.join(PATH, '.cache', key + '.pkl*')))


def fixed_split(lines: list) -> tuple:
    """
    (warmup A, runs A, runs B) of the given fixed-stage lines for two jobs on two nodes at the same time (short jobs
    start sooner on JUPITER): B gets only datasets whose standard HDF encoding already exists, so that it writes no
    cache and needs no warmup; A gets the warmup of the other datasets and their runs. The QM9 runs alternate between
    A and B (they take longest); the other runs of datasets with an existing encoding go to B.
    """
    warmup, runs_a, runs_b = [], [], []
    missing = {name for _, name, _, _ in GROUPS if not standard_hdf_cache_exists(name)}
    qm9 = 0
    for line in lines:
        name = params_of_line(line)['DATASET_NAME']
        if name in missing:
            runs_a.append(line)
        elif group_of_note_line(line) == 'qm9':
            (runs_a if qm9 % 2 == 0 else runs_b).append(line)
            qm9 += 1
        else:
            runs_b.append(line)
    warmup = [line for line in warmup_commands([group for group, *_ in GROUPS])
              if params_of_line(line)['DATASET_NAME'] in missing]
    return warmup, runs_a, runs_b


# rough minutes of a single-target run on a GPU, by group (JUPITER check 2323871: QM9 ~17 min per target + loading)
RUN_MINUTES = {'qm9': 20, 'compas': 7}


def opt_split() -> tuple:
    """
    (primer A, warmup A, studies A, studies B) of the opt stage for two jobs side by side: the HDF studies need the
    warmup (their encodings), so they run in job A after it; the fingerprint studies read only the dataset caches,
    which exist (JUPITER: copied from ex_13, including the 10% tuning subsamples of seed 420), and start at once in B.
    """
    primer, warmup = opt_warmup_commands()
    studies = opt_study_commands()
    return (primer, warmup, [line for line in studies if line.split()[2] == 'hdf'],
            [line for line in studies if line.split()[2] != 'hdf'])


def opteval_split() -> tuple:
    """
    (primer A, warmup A, runs A, runs B) of the opteval stage for two jobs side by side: job A runs the warmup (the
    encodings of the chosen HDF configurations) and all HDF runs; the fingerprint runs, which need no warmup, are
    assigned to the job with less expected work (A's HDF work counted first), the longest first.
    """
    primer, warmup, runs = opteval_commands()
    runs_a = [line for line in runs if 'predict_molecules__hdc.py' in line]
    load = {'a': sum(RUN_MINUTES.get(group_of_note_line(line), 1.5) for line in runs_a), 'b': 0.0}
    runs_b = []
    for line in runs:
        if 'predict_molecules__hdc.py' in line:
            continue
        job = min(load, key=load.get)
        (runs_a if job == 'a' else runs_b).append(line)
        load[job] += RUN_MINUTES.get(group_of_note_line(line), 1.5)
    return primer, warmup, runs_a, runs_b


def identity_of_line(line: str) -> tuple:
    """(method, NOTE, SEED) of a command line."""
    module = re.search(r'predict_molecules__(\w+)\.py', line).group(1)
    params = params_of_line(line)
    return method_of_params(module, params), params['NOTE'], params['SEED']


def identity_of_opt_line(line: str) -> tuple:
    """(method, NOTE, SEED) of a command line of any configuration (opteval)."""
    module = re.search(r'predict_molecules__(\w+)\.py', line).group(1)
    params = params_of_line(line)
    return method_of_module(module, params), params['NOTE'], params['SEED']


def completed(prefix: str) -> set:
    """(method, NOTE, SEED) of every completed archive (status 'done', no error) of the prefix."""
    done = set()
    for module in ('hdc', 'fp'):
        for meta_path in glob.glob(os.path.join(PATH, 'results', f'predict_molecules__{module}', f'{prefix}__*',
                                                'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
                continue
            done.add((method_of_params(module, params), params.get('NOTE'), params.get('SEED')))
    return done


def _write(name: str, lines: list):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'wrote {len(lines):4d} lines -> _ex27/{name}')


if __name__ == '__main__':
    if 'missing' in sys.argv[1:]:
        done = completed(PREFIX_FIXED)
        _write('fixed_missing.txt', [line for line in fixed_commands() if identity_of_line(line) not in done])
        sys.exit(0)
    if 'fixed_split' in sys.argv[1:]:
        # the not yet completed fixed-stage runs as two jobs (stages fixed_a and fixed_b, see fixed_split)
        done = completed(PREFIX_FIXED)
        split_warmup, split_a, split_b = fixed_split([line for line in fixed_commands()
                                                      if identity_of_line(line) not in done])
        _write('fixed_a_warmup.txt', split_warmup)
        _write('fixed_a_gpu.txt', split_a)
        _write('fixed_b_gpu.txt', split_b)
        sys.exit(0)
    if 'opteval' in sys.argv[1:]:
        opteval_primer, opteval_warmup, opteval_gpu = opteval_commands()
        _write('opteval_primer.txt', opteval_primer)
        _write('opteval_warmup.txt', opteval_warmup)
        _write('opteval_gpu.txt', opteval_gpu)
        sys.exit(0)
    if 'opt_full_init' in sys.argv[1:]:
        opt_full_init()
        sys.exit(0)
    if 'optfull_split' in sys.argv[1:]:
        for name, lines in zip(('optfull_a_primer', 'optfull_a_warmup', 'optfull_a_gpu', 'optfull_b_gpu'),
                               optfull_split()):
            _write(f'{name}.txt', lines)
        sys.exit(0)
    if 'opt_split' in sys.argv[1:]:
        for name, lines in zip(('opt_a_primer', 'opt_a_warmup', 'opt_a_gpu', 'opt_b_gpu'), opt_split()):
            _write(f'{name}.txt', lines)
        sys.exit(0)
    if 'opteval_split' in sys.argv[1:]:
        for name, lines in zip(('opteval_a_primer', 'opteval_a_warmup', 'opteval_a_gpu', 'opteval_b_gpu'),
                               opteval_split()):
            _write(f'{name}.txt', lines)
        sys.exit(0)
    if 'opteval_missing_split' in sys.argv[1:]:
        # the missing opteval runs as two jobs (stages opteval_missing_a / _b), alternating by line; the warmup of the
        # opteval stage has written every encoding they need
        done = completed_opteval()
        _, _, opteval_gpu = opteval_commands()
        missing = [line for line in opteval_gpu if identity_of_opt_line(line) not in done]
        _write('opteval_missing_a_gpu.txt', missing[0::2])
        _write('opteval_missing_b_gpu.txt', missing[1::2])
        sys.exit(0)
    if 'opteval_missing' in sys.argv[1:]:
        done = completed_opteval()
        _, _, opteval_gpu = opteval_commands()
        _write('opteval_missing.txt', [line for line in opteval_gpu if identity_of_opt_line(line) not in done])
        sys.exit(0)

    # every generated line must map back to its method (the analysis recognizes the methods by their parameters)
    fixed = fixed_commands()
    assert all(identity_of_line(line)[0] is not None for line in fixed)
    assert len(set(map(identity_of_line, fixed))) == len(fixed) == len(GROUPS) * len(METHODS) * len(SEEDS)

    check_gpu, check_cpu = check_commands()
    smoke = smoke_commands()
    check_groups = sorted({group_of_note(note) for note in CHECK_TARGETS} | set(SMOKE_GROUPS))
    _write('check_warmup.txt', warmup_commands(check_groups))
    _write('check_gpu.txt', check_gpu + smoke)
    _write('check_cpu.txt', check_cpu)
    _write('fixed_warmup.txt', warmup_commands([group for group, *_ in GROUPS]))
    _write('fixed_gpu.txt', fixed)
    opt_primer, opt_warmup = opt_warmup_commands()
    _write('opt_primer.txt', opt_primer)
    _write('opt_warmup.txt', opt_warmup)
    _write('opt_gpu.txt', opt_study_commands())
