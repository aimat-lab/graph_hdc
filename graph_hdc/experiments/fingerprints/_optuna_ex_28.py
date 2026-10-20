"""
One Optuna study of Experiment 28 (stage ex28_main of _slurm_ex_28.py): one model (gradient boosting, random forest or
KNN), one method (HDF, Morgan, RDKit) and one target of Figure 2.

As in ex_27 (_optuna_ex_27.py), TPE proposes the configurations and a trial runs predict_molecules__<module>.py with the
trial's representation and model parameters (_slurm_ex_28.suggest_params) on the tuning split (seed OPT_SEED, the full
dataset) as a subprocess; the objective is the MAE of the model on the run's held-out validation split (metrics
val_<model>), in the units of the target. The study is stored as a journal in OPT_DIR/<model>__<method>__<target>.log.

Differences to ex_27, because every search space of ex_28 is categorical (review 2026-10-20):

* The budget is OPT_TRIALS distinct configurations. TPE re-proposes evaluated configurations often in a categorical
  space (about a third of 30 trials in simulations); such a trial is not run again, since the run is deterministic
  (fixed seeds), but completed with the value of the earlier trial (user attribute 'duplicate_of'), which TPE sees as
  the same observation again. In ex_27, the continuous learning rate of the MLP made repeats practically impossible,
  so both experiments evaluate OPT_TRIALS distinct configurations per study.
* A trial that a job's time limit interrupted stays RUNNING in the journal. On the next start of the study it is marked
  as failed (user attribute 'stale') and its configuration is queued again, so that long configurations are not dropped
  in favor of the ones that fit into a job.

A trial whose run fails is recorded as failed and not counted; after MAX_FAILURES failed runs (stale trials excluded,
counted over all submissions) the study stops, and after MAX_TRIALS trials in total it stops as well.

    python _optuna_ex_28.py <model> <method> <target note> [number of configurations, default OPT_TRIALS; for tests]
"""
import os
import sys
import glob
import json
import shlex
import zlib
import subprocess

import optuna

from _slurm_ex_27 import PATH, METHOD
from _slurm_ex_28 import (OPT_DIR, OPT_TRIALS, PREFIX_TRIAL, MODELS_28, METHODS_28, NOTES_28, suggest_params,
                          trial_command, study_name, study_storage, config_key, distinct_complete)

MAX_FAILURES = 10
MAX_TRIALS = 10 * OPT_TRIALS
COMPLETE = optuna.trial.TrialState.COMPLETE


def run_trial(study: optuna.Study, model: str, method: str, note: str, trial: optuna.Trial) -> float:
    params = suggest_params(trial, model, method)
    trial.set_user_attr('params', json.dumps(params))
    earlier = [t for t in study.get_trials(deepcopy=False, states=(COMPLETE,))
               if t.user_attrs.get('params') and config_key(json.loads(t.user_attrs['params'])) == config_key(params)]
    if earlier:
        first = min(earlier, key=lambda t: t.number)
        trial.set_user_attr('duplicate_of', first.number)
        return first.value
    prefix = f'{PREFIX_TRIAL}__{model}__{method}__{note}__{trial.number}'
    trial.set_user_attr('prefix', prefix)
    line = trial_command(prefix, model, method, note, params)
    # the archive of this run is the one folder of the prefix that is new afterwards
    pattern = os.path.join(PATH, 'results', f'predict_molecules__{METHOD[method][0]}', f'{prefix}__*',
                           'experiment_data.json')
    before = set(glob.glob(pattern))
    result = subprocess.run(shlex.split(line), cwd=PATH, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True)
    new = set(glob.glob(pattern)) - before
    if result.returncode != 0 or len(new) != 1:
        trial.set_user_attr('failed_run', True)
        raise RuntimeError(f'trial run {prefix} failed (exit code {result.returncode}, {len(new)} new archives): '
                           f'{result.stderr[-3000:]}')
    data = json.load(open(new.pop()))
    return float(data['metrics'][f'val_{model}']['mae'])


def requeue_stale(study: optuna.Study, name: str):
    """Mark the RUNNING trials of an earlier, interrupted process as failed and queue their configurations again."""
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study._storage.set_trial_user_attr(trial._trial_id, 'stale', True)
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        if trial.params:
            study.enqueue_trial(trial.params, skip_if_exists=False)
        print(f'{name}: trial {trial.number} was interrupted; its configuration is queued again: {trial.params}')


def main(model: str, method: str, note: str, n_trials: int = OPT_TRIALS):
    assert model in MODELS_28 and method in METHODS_28 and note in NOTES_28, (model, method, note)
    os.makedirs(OPT_DIR, exist_ok=True)
    name = study_name(model, method, note)
    seed = zlib.crc32(name.encode())
    study = optuna.create_study(study_name=name, storage=study_storage(model, method, note), direction='minimize',
                                load_if_exists=True)
    requeue_stale(study, name)
    while True:
        # a new sampler for every trial, seeded with the study and its number of trials so far (as in ex_27)
        study.sampler = optuna.samplers.TPESampler(seed=(seed + len(study.trials)) % 2 ** 32)
        trials = study.get_trials(deepcopy=False)
        distinct = len(distinct_complete(trials))
        failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL and t.user_attrs.get('failed_run'))
        if distinct >= n_trials:
            print(f'{name}: {distinct} distinct configurations in {len(trials)} trials, best validation MAE '
                  f'{study.best_value:.6g} with {study.best_trial.user_attrs["params"]}')
            return
        if failed >= MAX_FAILURES:
            sys.exit(f'{name}: {failed} failed runs, stopping')
        if len(trials) >= MAX_TRIALS:
            sys.exit(f'{name}: {len(trials)} trials but only {distinct} distinct configurations, stopping')
        study.optimize(lambda trial: run_trial(study, model, method, note, trial), n_trials=1,
                       catch=(RuntimeError, KeyError, ValueError))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(n) for n in sys.argv[4:5]))
