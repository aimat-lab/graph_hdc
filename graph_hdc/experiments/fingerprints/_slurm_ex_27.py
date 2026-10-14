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
  ex_27_trial__<method>__<target>__<number>) with the trial's parameters on the tuning split: SEED OPT_SEED (as the
  tuning seed of ex_13, not one of the evaluation seeds), QM9 and COMPAS-3X subsampled to 10% (NUM_DATA 0.1), MLP only;
  the objective is its validation MAE. Search spaces: see suggest_params (user decisions 2026-10-13).
  Files: opt_warmup.txt (every HDF encoding the trials can ask for), opt_gpu.txt (the studies)
* ``opteval`` (prefix ex_27_opt): the best configuration of every study, evaluated over the seeds 0-9 on the full data
  (MLP only; one run per target, as the configurations differ). Written by ``python _slurm_ex_27.py opteval`` after the
  studies: opteval_warmup.txt (the HDF encodings of the chosen configurations), opteval_gpu.txt

The warmups encode the datasets once with the fixed codebook (short runs without models), so that the parallel runs
only read the HDF encoding caches, which none of them could write without racing the others. The runner stops if a
warmup run fails, because pycomex writes its caches in place and a failed run can leave a truncated cache file behind;
delete that file before the next attempt.

Usage (from the fingerprints folder of the ex_27 worktree on JUPITER):
    python _slurm_ex_27.py              # writes all command files
    sbatch --export=ALL,EX27_STAGE=check --time=04:00:00 run_ex27.sbatch
    python analyze_ex_27.py check
    sbatch --export=ALL,EX27_STAGE=fixed run_ex27.sbatch
    python _slurm_ex_27.py missing      # after an interrupted fixed stage (no ex_27 job running): fixed_missing.txt
    sbatch --export=ALL,EX27_STAGE=fixed,GPU_FILE=fixed_missing.txt run_ex27.sbatch
    sbatch --export=ALL,EX27_STAGE=opt,GPU_J=88 run_ex27.sbatch     # an interrupted study resumes when resubmitted
    python _slurm_ex_27.py opteval      # after all studies are complete
    sbatch --export=ALL,EX27_STAGE=opteval run_ex27.sbatch
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
PREFIX_TRIAL = 'ex_27_trial'
PREFIX_OPT = 'ex_27_opt'
OPT_TRIALS = 30
OPT_SEED = 420
# the groups whose datasets are subsampled to 10% on the tuning split (as in ex_13)
OPT_SUBSAMPLED = {'qm9': 0.1, 'compas': 0.1}
OPT_DIR = os.path.join(OUT, 'optuna')
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


def opt_warmup_commands() -> list:
    """Every HDF encoding the trials can ask for (size x depth x graph size and diameter) on the tuning splits."""
    lines, seen = [], set()
    for group, name, _notes, _extra in GROUPS:
        if name in seen:
            continue
        seen.add(name)
        for size in reversed(SIZES):
            for depth in HDF_DEPTHS:
                for attributes in (True, False):
                    encoder = {'EMBEDDING_SIZE': size, 'NUM_LAYERS': depth, 'GRAPH_ATTRIBUTES': attributes}
                    lines.append((group, hdf_warmup_command(name, OPT_SEED, opt_num_data(plain_group(name)), encoder)))
    return [line for group, line in sorted(lines, key=lambda gl: -expected_cost(gl[0]))]


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
    """The best complete trial of a study (raises if the study has fewer than OPT_TRIALS complete trials)."""
    import optuna
    study = optuna.load_study(study_name=f'{method}__{note}', storage=study_storage(method, note))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < OPT_TRIALS:
        raise RuntimeError(f'study {method}__{note} has {len(complete)} of {OPT_TRIALS} complete trials')
    return min(complete, key=lambda t: t.value)


def best_params(method: str, note: str) -> dict:
    """The run parameters of the best trial of a study (stored by _optuna_ex_27.py as the user attribute 'params')."""
    params = json.loads(best_trial(method, note).user_attrs['params'])
    params['NN_HIDDEN_LAYER_SIZES'] = tuple(params['NN_HIDDEN_LAYER_SIZES'])
    return params


def opteval_commands() -> tuple:
    """(warmup lines, GPU lines) of the evaluation of the best configurations over the seeds on the full data."""
    warmup, runs, seen = [], [], set()
    for method, module, *_ in METHODS:
        for group, name, notes, _extra in GROUPS:
            for note in notes:
                params = best_params(method, note)
                if module == 'hdc':
                    encoder = {k: params[k] for k in ('EMBEDDING_SIZE', 'NUM_LAYERS', 'GRAPH_ATTRIBUTES')}
                    if (name, *encoder.values()) not in seen:
                        seen.add((name, *encoder.values()))
                        warmup.append(hdf_warmup_command(name, 0, 1.0, encoder))
                for seed in SEEDS:
                    runs.append((group, _command(module, PREFIX_OPT, seed, group, [note],
                                                 {**params, 'MODELS': ['neural_net2'], 'NN_DEVICE_DATA': True})))
    return warmup, [line for group, line in sorted(runs, key=lambda gl: -expected_cost(gl[0]))]


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


def identity_of_line(line: str) -> tuple:
    """(method, NOTE, SEED) of a command line."""
    module = re.search(r'predict_molecules__(\w+)\.py', line).group(1)
    params = params_of_line(line)
    return method_of_params(module, params), params['NOTE'], params['SEED']


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
    if 'opteval' in sys.argv[1:]:
        opteval_warmup, opteval_gpu = opteval_commands()
        _write('opteval_warmup.txt', opteval_warmup)
        _write('opteval_gpu.txt', opteval_gpu)
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
    _write('opt_warmup.txt', opt_warmup_commands())
    _write('opt_gpu.txt', opt_study_commands())
