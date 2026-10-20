"""
Local smoke test of ex_28 (before deploying to JUPITER): small runs in the exact command format of _slurm_ex_28, on
small datasets, which exercise every new code path, and a check of their archives.

1. random forest + gradient boosting with the fixed configurations, two targets in one run (HOPV15, TARGET_INDICES)
2. HDF + KNN with the torch backend, cosine distance and distance weights (FreeSolv)
3. random forest with 10% features per split and unlimited depth (FreeSolv, RDKit)
4. gradient boosting with learning rate 0.2 and depth 4 (FreeSolv, HDF)
5. the study driver _optuna_ex_28.py with 2 trials of the KNN on Morgan fingerprints for ClogP, into a temporary
   journal folder (OPT_DIR is redirected)

    python _smoke_ex_28.py         (from the fingerprints folder; prefix ex_28_smoke, removes nothing)
"""
import os
import sys
import glob
import json
import shlex
import subprocess

import _slurm_ex_28 as s28
from _slurm_ex_27 import PATH, GROUP, METHOD, HDF, TARGET_INDEX, _command

PREFIX = 'ex_28_smoke'


def run(line: str, module: str) -> dict:
    pattern = os.path.join(PATH, 'results', f'predict_molecules__{module}', f'{PREFIX}__*', 'experiment_data.json')
    before = set(glob.glob(pattern))
    result = subprocess.run(shlex.split(line), cwd=PATH, capture_output=True, text=True)
    new = set(glob.glob(pattern)) - before
    if result.returncode != 0 or len(new) != 1:
        print(result.stderr[-3000:])
        raise SystemExit(f'FAILED: exit {result.returncode}, {len(new)} archives: {line[:200]}')
    return json.load(open(new.pop()))


def check(name: str, data: dict, keys: list):
    metrics = data.get('metrics', {})
    for key in keys:
        values = metrics.get(key, {})
        assert 'r2' in values and 'mae' in values, f'{name}: no metrics {key} (have {sorted(metrics)})'
        print(f'   {name}: {key} r2 {values["r2"]:.3f} mae {values["mae"]:.4g}')


fixed_models = {**s28.FIXED, 'MODELS': ['random_forest', 'grad_boost'], 'RF_NUM_JOBS': 4}
notes = GROUP['hopv15'][1]
line = _command('fp', PREFIX, 0, 'hopv15', notes, {**METHOD['morgan'][1], **fixed_models})
data = run(line, 'fp')
check('1 RF+GB two targets', data, [f'{split}_{model}__t{TARGET_INDEX[note]}' for note in notes
                                   for model in ('random_forest', 'grad_boost') for split in ('test', 'val')])

line = _command('hdc', PREFIX, 0, 'freesolv', ['freesolv_hfe'],
                {**HDF, 'MODELS': ['k_neighbors'], 'KN_BACKEND': 'torch', 'KN_METRIC': 'cosine',
                 'KN_NUM_NEIGHBORS': 5, 'KN_WEIGHTS': 'distance'})
check('2 HDF KNN distance', run(line, 'hdc'), ['test_k_neighbors', 'val_k_neighbors'])

line = _command('fp', PREFIX, 0, 'freesolv', ['freesolv_hfe'],
                {**METHOD['rdkit'][1], 'MODELS': ['random_forest'], 'RF_NUM_ESTIMATORS': 10, 'RF_MAX_DEPTH': None,
                 'RF_MAX_FEATURES': 0.1, 'RF_NUM_JOBS': 4})
check('3 RF 10% features', run(line, 'fp'), ['test_random_forest', 'val_random_forest'])

line = _command('hdc', PREFIX, 0, 'freesolv', ['freesolv_hfe'],
                {**HDF, 'MODELS': ['grad_boost'], 'GB_NUM_ESTIMATORS': 10, 'GB_MAX_DEPTH': 4, 'GB_LEARNING_RATE': 0.2})
check('4 GB lr 0.2', run(line, 'hdc'), ['test_grad_boost', 'val_grad_boost'])

# 5: the study driver, with its journals in a temporary folder
import tempfile  # noqa: E402
import optuna  # noqa: E402
import _optuna_ex_28 as o28  # noqa: E402
tmp = tempfile.mkdtemp(prefix='ex28_smoke_optuna_')
s28.OPT_DIR = o28.OPT_DIR = tmp
o28.PREFIX_TRIAL = PREFIX + '_trial'
o28.main('k_neighbors', 'morgan', 'clogp', n_trials=2)
storage = s28.study_storage('k_neighbors', 'morgan', 'clogp')
study = optuna.load_study(study_name='k_neighbors__morgan__clogp', storage=storage)
assert len(s28.distinct_complete(study.trials)) == 2
print(f'   5 study driver: 2 distinct configurations, values {[round(t.value, 4) for t in study.trials]}')

# 6: an interrupted trial (RUNNING, as after a job's time limit) and a repeated configuration
pattern = os.path.join(PATH, 'results', 'predict_molecules__fp', f'{PREFIX}_trial__*', 'experiment_data.json')
archives_before = len(glob.glob(pattern))
used = {s28.config_key(json.loads(t.user_attrs['params'])) for t in study.trials}
stale_params = next(p for p in ({'size': size, 'radius': radius, 'kn_neighbors': k, 'kn_weights': w}
                                for size in (2048, 1024) for radius in (1, 3) for k in (10, 3) for w in ('uniform',))
                    if s28.config_key(s28.suggest_params(optuna.trial.FixedTrial(p), 'k_neighbors', 'morgan'))
                    not in used)
study.enqueue_trial(stale_params)
stale = study.ask()
s28.suggest_params(stale, 'k_neighbors', 'morgan')  # left RUNNING
study.enqueue_trial(study.trials[0].params)          # a repetition of trial 0
o28.main('k_neighbors', 'morgan', 'clogp', n_trials=4)
study = optuna.load_study(study_name='k_neighbors__morgan__clogp', storage=storage)
trials = study.trials
states = [(t.number, t.state.name, t.user_attrs.get('duplicate_of'), t.user_attrs.get('stale')) for t in trials]
assert any(t.user_attrs.get('stale') and t.state == optuna.trial.TrialState.FAIL for t in trials), states
assert sum(1 for t in trials if t.user_attrs.get('duplicate_of') == 0) == 1, states
assert any(t.state == optuna.trial.TrialState.COMPLETE and t.params == stale_params for t in trials), states
assert len(s28.distinct_complete(trials)) == 4, states
new_archives = len(glob.glob(pattern)) - archives_before
assert new_archives == 2, f'{new_archives} new archives (expected: the requeued configuration and one new one)'
print(f'   6 interrupted + repeated trials: {states}; {new_archives} new runs; journal in {tmp}')
print('SMOKE OK')
