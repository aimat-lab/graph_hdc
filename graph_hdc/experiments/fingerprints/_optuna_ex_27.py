"""
One Optuna study of the optimized table of Experiment 27 (stage ``opt`` of _slurm_ex_27.py): one method and one target.

Every method gets the same budget (user decisions 2026-10-13): OPT_TRIALS complete trials of TPE per method and target.
A trial runs predict_molecules__<module>.py with the trial's representation and MLP parameters (suggest_params) on the
tuning split (seed OPT_SEED, not one of the evaluation seeds; the full dataset, see OPT_SUBSAMPLED), MLP only, as a
subprocess on the GPU of the calling slot, and reads the objective from the run's archive: the MAE on the run's
held-out validation split (metrics val_neural_net2), in the units of the target. The study is stored as a journal in
OPT_DIR/<method>__<target>.log (round 2: _ex27/optuna_full), so that a resubmitted study continues with its complete
trials. A trial whose run fails is recorded as failed and not counted; after MAX_FAILURES failed trials (counted over
all submissions) the study stops.

    python _optuna_ex_27.py <method> <target note> [number of trials, default OPT_TRIALS; for tests]
"""
import os
import sys
import glob
import json
import shlex
import zlib
import subprocess

import optuna

from _slurm_ex_27 import (PATH, OPT_DIR, OPT_TRIALS, PREFIX_TRIAL, METHOD, suggest_params, opt_trial_command,
                          study_storage, group_of_note)

MAX_FAILURES = 10


def run_trial(method: str, note: str, trial: optuna.Trial) -> float:
    params = suggest_params(trial, method)
    prefix = f'{PREFIX_TRIAL}__{method}__{note}__{trial.number}'
    trial.set_user_attr('params', json.dumps(params))
    trial.set_user_attr('prefix', prefix)
    line = opt_trial_command(prefix, method, note, params)
    # the archive of this run is the one folder of the prefix that is new afterwards (an earlier attempt of a study
    # whose journal was deleted can have left archives with the same prefix)
    pattern = os.path.join(PATH, 'results', f'predict_molecules__{METHOD[method][0]}', f'{prefix}__*',
                           'experiment_data.json')
    before = set(glob.glob(pattern))
    result = subprocess.run(shlex.split(line), cwd=PATH, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True)
    new = set(glob.glob(pattern)) - before
    if result.returncode != 0 or len(new) != 1:
        raise RuntimeError(f'trial run {prefix} failed (exit code {result.returncode}, {len(new)} new archives): '
                           f'{result.stderr[-3000:]}')
    data = json.load(open(new.pop()))
    return float(data['metrics']['val_neural_net2']['mae'])


def main(method: str, note: str, n_trials: int = OPT_TRIALS):
    assert method in METHOD and group_of_note(note)
    os.makedirs(OPT_DIR, exist_ok=True)
    name = f'{method}__{note}'
    seed = zlib.crc32(name.encode())
    study = optuna.create_study(study_name=name, storage=study_storage(method, note), direction='minimize',
                                load_if_exists=True)
    while True:
        # A new sampler for every trial, seeded with the study and its number of trials so far: reproducible for a
        # given history, and a resumed study does not repeat the random start-up configurations of its first trials
        # (TPE keeps no state besides its random generator).
        study.sampler = optuna.samplers.TPESampler(seed=(seed + len(study.trials)) % 2 ** 32)
        states = [t.state for t in study.trials]
        complete = states.count(optuna.trial.TrialState.COMPLETE)
        failed = states.count(optuna.trial.TrialState.FAIL)
        if complete >= n_trials:
            print(f'{name}: {complete} complete trials, best validation MAE {study.best_value:.6g} '
                  f'with {study.best_trial.user_attrs["params"]}')
            return
        if failed >= MAX_FAILURES:
            sys.exit(f'{name}: {failed} failed trials, stopping')
        study.optimize(lambda trial: run_trial(method, note, trial), n_trials=1,
                       catch=(RuntimeError, KeyError, ValueError))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], *(int(n) for n in sys.argv[3:4]))
