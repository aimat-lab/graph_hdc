"""
Command-file generator for Experiment 28 (ex_28): the model comparison of Figure 2 of the main text with the current
HDF encoder, on JUPITER (reviewer comment R2.5; user decisions 2026-10-19 and 2026-10-20).

Figure 2 compares four model types (gradient boosting, KNN, random forest, neural network) on HDF, Morgan and RDKit
fingerprints for Crippen's logP, AqSolDB logS and the QM9 internal energy U0 and HOMO-LUMO gap, each with the fixed
standard configuration and after tuning with the same budget for every representation. The neural network (fixed and
tuned) and the fixed KNN come from ex_27, which uses the same splits (the JUPITER dataset caches), the same HDF codebook
and the same tuning protocol; ex_28 runs only the rest ("only the ones that are missing", user):

1. ``fixed`` (prefix ex_28_fixed): gradient boosting and random forest with the fixed configurations of the SI
   (tab:ml_hyperparams) on the standard representations of ex_27, both models in one run (QM9 with both targets via
   TARGET_INDICES): 3 dataset groups x 3 methods x 10 seeds = 90 runs.
2. ``opt``: one Optuna study (TPE) per model (gradient boosting, random forest, KNN), method and target, 36 studies
   (_optuna_ex_28.py), as in ex_27: the tuning seed OPT_SEED, the full data, objective the MAE on the validation split,
   a budget of OPT_TRIALS distinct configurations (see _optuna_ex_28.py on repeated configurations). The representation
   spaces are those of ex_27; the model spaces (user, 2026-10-20) are
       random forest      trees {10, 100, 500}, depth {None, 10, 20}, features per split {sqrt(d), 10%}
       gradient boosting  estimators {10, 100, 300}, depth {2, 3, 4}, learning rate {0.01, 0.1, 0.2}
       KNN                k {3, 5, 10}, weights {uniform, distance}, the native metric of each representation
   (all features per split, as in the original grid, would cost about 90 CPU hours per QM9 trial).
3. ``opteval`` (prefix ex_28_opt): the best of the first OPT_TRIALS distinct configurations of every study, evaluated
   over the seeds 0-9: 360 runs.

Pools on a Booster node (run_ex27.sbatch with EX27_STAGE=ex28_main / ex28_opteval). JUPITER Booster jobs take whole
nodes and production jobs must use all 4 GPUs, so every stage combines both pools: the CPU pool (CPU_J x CPU_THREADS
cores) runs the random forests (RF_NUM_JOBS = CPU_THREADS) and, in ex28_main, after the random-forest studies the fixed
runs; the GPU pool runs the KNN (exact search with the torch backend, on a GPU) and gradient boosting (single-threaded,
GPU unused). The HDF encodings of every configuration the runs can ask for exist from ex_27 (stages opt and optfull);
the generator writes warmup lines (run first, as a barrier) only for missing ones.

Usage (from the fingerprints folder of the ex_27 worktree on JUPITER; never regenerate while an ex_28 job runs):
    python _slurm_ex_28.py            # ex28_main_*: all 36 studies and the fixed runs without a completed archive
    sbatch --export=ALL,EX27_STAGE=ex28_main,CPU_J=12,CPU_THREADS=20,GPU_J=24 --time=04:00:00 run_ex27.sbatch
    (resubmit, after regenerating, until all fixed runs and studies are complete: studies continue from their journals,
    interrupted trials first; for the last long random-forest studies, EX28_RF_THREADS=48 python _slurm_ex_28.py and
    CPU_J=5,CPU_THREADS=48 give each study more threads)
    python _slurm_ex_28.py opteval    # after all studies: ex28_opteval_*, the evaluation runs without an archive
    sbatch --export=ALL,EX27_STAGE=ex28_opteval,CPU_J=12,CPU_THREADS=20,GPU_J=24 --time=04:00:00 run_ex27.sbatch
    python analyze_ex_28.py
"""
import os
import sys
import glob
import json

from _slurm_ex_14 import params_of_line
from _slurm_ex_27 import (PATH, OUT, GROUP, METHOD, SEEDS, OPT_SEED, SIZES, HDF_DEPTHS, CODEBOOK_SEED, HDF, _command,
                          group_of_note, hdf_warmup_command, split_primer, method_of_module, matches_params)

PREFIX_FIXED = 'ex_28_fixed'
PREFIX_TRIAL = 'ex_28_trial'
PREFIX_OPT = 'ex_28_opt'
OPT_TRIALS = 30
OPT_DIR = os.path.join(OUT, 'optuna_ex28')

METHODS_28 = ['hdf', 'morgan', 'rdkit']
NOTES_28 = ['clogp', 'aqsoldb_logs', 'qm9_energy', 'qm9_gap']
MODELS_28 = ['grad_boost', 'random_forest', 'k_neighbors']
# the dataset groups of _slurm_ex_27 and the targets of Figure 2 in them (QM9: one run for both targets)
FIXED_GROUPS = {'qm9': ['qm9_energy', 'qm9_gap'], 'aqsoldb': ['aqsoldb_logs'], 'clogp': ['clogp']}
assert sorted(n for notes in FIXED_GROUPS.values() for n in notes) == sorted(NOTES_28)

# the fixed configurations of the SI (tab:ml_hyperparams); the code of gradient boosting additionally uses subsample
# 0.75 and sqrt(d) features per split in every configuration
FIXED = {'RF_NUM_ESTIMATORS': 100, 'RF_MAX_DEPTH': 10, 'RF_MAX_FEATURES': 'sqrt',
         'GB_NUM_ESTIMATORS': 100, 'GB_LEARNING_RATE': 0.01, 'GB_MAX_DEPTH': 3}
# The pools of every stage: CPU_J random-forest slots with CPU_THREADS threads each and GPU_J slots for gradient
# boosting and KNN with 2 threads each (run_ex27.sbatch): 12 x 20 + 24 x 2 = 288 cores. The thread count of a random
# forest is an execution setting, not part of a configuration: the trials of a study take it from OMP_NUM_THREADS, which
# the CPU pool sets to CPU_THREADS; the fixed and opteval lines get EX28_RF_THREADS (default 20) when they are written,
# so a later round can use other pool sizes (e.g. CPU_J=6, CPU_THREADS=48 for the last long studies).
RF_THREADS = int(os.environ.get('EX28_RF_THREADS', 20))
CPU_J, GPU_J = max(1, 240 // RF_THREADS), 24
SUBMIT = f'CPU_J={CPU_J},CPU_THREADS={RF_THREADS},GPU_J={GPU_J}'


def search_space(model: str, method: str) -> list:
    """
    The search space of a study as [(Optuna parameter name, choices, run parameter)]: the representation space of
    ex_27, then the model space (see the docstring). Every parameter is categorical.
    """
    if method == 'hdf':
        space = [('size', SIZES, 'EMBEDDING_SIZE'), ('depth', HDF_DEPTHS, 'NUM_LAYERS'),
                 ('global_descriptors', [True, False], 'GRAPH_ATTRIBUTES')]
    elif method == 'morgan':
        space = [('size', SIZES, 'FINGERPRINT_SIZE'), ('radius', [1, 2, 3], 'FINGERPRINT_RADIUS')]
    elif method == 'rdkit':
        # predict_molecules__fp builds RDKit fingerprints from paths of up to 2 x FINGERPRINT_RADIUS bonds
        space = [('size', SIZES, 'FINGERPRINT_SIZE'), ('max_path_half', [1, 2, 3, 4], 'FINGERPRINT_RADIUS')]
    else:
        raise ValueError(method)
    if model == 'random_forest':
        space += [('rf_estimators', [10, 100, 500], 'RF_NUM_ESTIMATORS'), ('rf_depth', [None, 10, 20], 'RF_MAX_DEPTH'),
                  ('rf_features', ['sqrt', 0.1], 'RF_MAX_FEATURES')]
    elif model == 'grad_boost':
        space += [('gb_estimators', [10, 100, 300], 'GB_NUM_ESTIMATORS'), ('gb_depth', [2, 3, 4], 'GB_MAX_DEPTH'),
                  ('gb_learning_rate', [0.01, 0.1, 0.2], 'GB_LEARNING_RATE')]
    elif model == 'k_neighbors':
        space += [('kn_neighbors', [3, 5, 10], 'KN_NUM_NEIGHBORS'), ('kn_weights', ['uniform', 'distance'], 'KN_WEIGHTS')]
    else:
        raise ValueError(model)
    return space


def suggest_params(trial, model: str, method: str) -> dict:
    """The run parameters of one trial: the fixed parameters of the method and one choice per search parameter."""
    params = dict(METHOD[method][1])
    for name, choices, key in search_space(model, method):
        params[key] = trial.suggest_categorical(name, choices)
    return params


def model_params(model: str, method: str, rf_threads: int = RF_THREADS) -> dict:
    """Run parameters that every run of a model gets: the model itself and its execution settings."""
    params = {'MODELS': [model]}
    if model == 'random_forest':
        params['RF_NUM_JOBS'] = rf_threads
    if model == 'k_neighbors':
        params.update({'KN_BACKEND': 'torch', 'KN_METRIC': METHOD[method][2]})
    return params


# parameters that only affect how a run executes, not its result: ignored when comparing configurations and runs
EXECUTION_KEYS = ('RF_NUM_JOBS',)


def config_key(params: dict) -> str:
    """A canonical string of a configuration (the parameters of a trial), for finding repeated configurations."""
    return json.dumps({k: v for k, v in params.items() if k not in EXECUTION_KEYS}, sort_keys=True)


def distinct_complete(trials) -> list:
    """The complete trials that are the first evaluation of their configuration, in the order of their numbers."""
    import optuna
    seen, out = set(), []
    for trial in sorted(trials, key=lambda t: t.number):
        if trial.state != optuna.trial.TrialState.COMPLETE or 'params' not in trial.user_attrs:
            continue
        key = config_key(json.loads(trial.user_attrs['params']))
        if key not in seen:
            seen.add(key)
            out.append(trial)
    return out


def trial_command(prefix: str, model: str, method: str, note: str, params: dict) -> str:
    """A trial: the trial's parameters on the tuning split (seed OPT_SEED, full data), one model. A random forest uses
    the threads of the pool slot the study runs in (OMP_NUM_THREADS)."""
    threads = int(os.environ.get('OMP_NUM_THREADS') or RF_THREADS)
    return _command(METHOD[method][0], prefix, OPT_SEED, group_of_note(note), [note],
                    {**params, **model_params(model, method, threads), 'NUM_DATA': 1.0})


def study_name(model: str, method: str, note: str) -> str:
    return f'{model}__{method}__{note}'


def study_storage(model: str, method: str, note: str):
    import optuna
    path = os.path.join(OPT_DIR, f'{study_name(model, method, note)}.log')
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path))


def best_trial(model: str, method: str, note: str):
    """
    The best of the first OPT_TRIALS distinct configurations of a study (their first evaluations, by trial number; see
    _optuna_ex_28.py on repeated configurations). Raises if the study has fewer.
    """
    import optuna
    name = study_name(model, method, note)
    if not os.path.exists(os.path.join(OPT_DIR, f'{name}.log')):
        raise RuntimeError(f'study {name} does not exist')
    study = optuna.load_study(study_name=name, storage=study_storage(model, method, note))
    distinct = distinct_complete(study.trials)
    if len(distinct) < OPT_TRIALS:
        raise RuntimeError(f'study {name} has {len(distinct)} of {OPT_TRIALS} distinct complete configurations')
    return min(distinct[:OPT_TRIALS], key=lambda t: t.value)


def best_params(model: str, method: str, note: str) -> dict:
    """The run parameters of the best trial of a study (stored by _optuna_ex_28.py as the user attribute 'params')."""
    return json.loads(best_trial(model, method, note).user_attrs['params'])


def hdf_cache_exists(name: str, size: int, depth: int, attributes: bool) -> bool:
    key = (f'hdc_{name}__codebook_{CODEBOOK_SEED}__size_{size}__depth_{depth}__bidir__totalh__unitspec'
           + ('' if attributes else '__noglobal'))
    return bool(glob.glob(os.path.join(PATH, '.cache', key + '.pkl*')))


def missing_hdf_warmups(configurations) -> tuple:
    """(primer, rest) warmup lines for the (DATASET_NAME, size, depth, attributes) whose encoding is not cached."""
    lines = [hdf_warmup_command(name, 0, 1.0, {'EMBEDDING_SIZE': size, 'NUM_LAYERS': depth,
                                               'GRAPH_ATTRIBUTES': attributes})
             for name, size, depth, attributes in sorted(set(configurations), key=lambda c: (-c[1], c))
             if not hdf_cache_exists(name, size, depth, attributes)]
    return split_primer(lines)


def all_hdf_configurations() -> list:
    names = {GROUP[group_of_note(note)][0] for note in NOTES_28}
    return [(name, size, depth, attributes) for name in names for size in SIZES for depth in HDF_DEPTHS
            for attributes in (True, False)]


def expected_cost(line: str) -> int:
    """Rough cost class of a line (higher = longer), to start the QM9 lines first."""
    return 1 if "qm9_smiles" in line else 0


def fixed_commands() -> list:
    lines = []
    for group, notes in FIXED_GROUPS.items():
        for method in METHODS_28:
            module, feat, _metric = METHOD[method]
            for seed in SEEDS:
                lines.append(_command(module, PREFIX_FIXED, seed, group, notes,
                                      {**feat, **FIXED, 'MODELS': ['random_forest', 'grad_boost'],
                                       'RF_NUM_JOBS': RF_THREADS}))
    return sorted(lines, key=lambda line: -expected_cost(line))


def study_commands(models) -> list:
    """One line per study, the QM9 studies (the longest; HDF first, whose continuous features make trees costliest)
    first."""
    lines = [f'python _optuna_ex_28.py {model} {method} {note}'
             for model in models for method in METHODS_28 for note in NOTES_28]
    return sorted(lines, key=lambda line: (-int('qm9' in line), -int(' hdf ' in line)))


def interleave(heavy: list, light: list) -> list:
    """The lines of both lists alternately (the rest of the longer list at the end), so that the QM9 runs, which need
    most memory, do not all start at the same time."""
    out = []
    for k in range(max(len(heavy), len(light))):
        out += heavy[k:k + 1] + light[k:k + 1]
    return out


def opteval_commands() -> tuple:
    """(RF lines for the CPU pool, gradient boosting and KNN lines for the GPU pool), QM9 and the other datasets
    alternating (memory: a QM9 run at 8192 dimensions needs up to ~25 GB, the node 480 GB of CPU memory)."""
    cpu, gpu = {True: [], False: []}, {True: [], False: []}
    for model in MODELS_28:
        for method in METHODS_28:
            module = METHOD[method][0]
            for note in NOTES_28:
                params = best_params(model, method, note)
                for seed in SEEDS:
                    line = _command(module, PREFIX_OPT, seed, group_of_note(note), [note],
                                    {**params, **model_params(model, method)})
                    pool = cpu if model == 'random_forest' else gpu
                    pool[expected_cost(line) > 0].append(line)
    return interleave(cpu[True], cpu[False]), interleave(gpu[True], gpu[False])


def opteval_configurations() -> list:
    out = []
    for model in MODELS_28:
        for note in NOTES_28:
            params = best_params(model, 'hdf', note)
            out.append((GROUP[group_of_note(note)][0], params['EMBEDDING_SIZE'], params['NUM_LAYERS'],
                        params['GRAPH_ATTRIBUTES']))
    return out


def completed(prefix: str) -> list:
    """The parameters of every completed archive (status done, no error) of the prefix."""
    done = []
    for module in ('hdc', 'fp'):
        for meta_path in glob.glob(os.path.join(PATH, 'results', f'predict_molecules__{module}', f'{prefix}__*',
                                                'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            if params.get('__PREFIX__') == prefix and meta.get('status') == 'done' and not meta.get('has_error'):
                done.append((module, params))
    return done


def line_is_done(line: str, done: list) -> bool:
    """Whether a command line has a completed archive with the same parameters (same module, all line parameters)."""
    module = 'hdc' if 'predict_molecules__hdc.py' in line else 'fp'
    # the special pycomex parameters (__DEBUG__, __INCLUDE__) are not stored with the others, except the prefix; the
    # execution settings (thread counts) do not change a result
    params = {k: v for k, v in params_of_line(line).items()
              if (not k.startswith('__') or k == '__PREFIX__') and k not in EXECUTION_KEYS}
    return any(m == module and matches_params(p, params) for m, p in done)


def _write(name: str, lines: list):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'wrote {len(lines):4d} lines -> _ex27/{name}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'main'
    if mode == 'opteval':
        # the evaluation runs without a completed archive (all of them on the first call)
        done = completed(PREFIX_OPT)
        primer, warmup = missing_hdf_warmups(opteval_configurations())
        cpu, gpu = opteval_commands()
        _write('ex28_opteval_primer.txt', primer)
        _write('ex28_opteval_warmup.txt', warmup)
        _write('ex28_opteval_cpu.txt', [line for line in cpu if not line_is_done(line, done)])
        _write('ex28_opteval_gpu.txt', [line for line in gpu if not line_is_done(line, done)])
        print(f'submit stage ex28_opteval with {SUBMIT}')
        sys.exit(0)
    if mode != 'main':
        sys.exit(f'unknown mode {mode}')

    fixed = fixed_commands()
    assert len(fixed) == len(FIXED_GROUPS) * len(METHODS_28) * len(SEEDS)
    done = completed(PREFIX_FIXED)
    fixed = [line for line in fixed if not line_is_done(line, done)]
    primer, warmup = missing_hdf_warmups(all_hdf_configurations())
    _write('ex28_main_primer.txt', primer)
    _write('ex28_main_warmup.txt', warmup)
    # CPU pool: the random-forest studies first (the QM9 ones take longest), then the fixed runs, which fill the slots
    # of the studies that finish early; GPU pool: KNN and gradient boosting. A study that is already complete only
    # loads its journal and exits.
    _write('ex28_main_cpu.txt', study_commands(['random_forest']) + fixed)
    _write('ex28_main_gpu.txt', study_commands(['k_neighbors', 'grad_boost']))
    print(f'{len(fixed)} fixed runs without an archive; submit stage ex28_main with {SUBMIT}')
