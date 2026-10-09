"""
Command-file generator for Experiment 19 (ex_19): HDF vs. modern molecular fingerprints on the splits of the GNN
comparison (reviewer comment R2.1). Successor of ex_18, which compared HDF with Morgan, SECFP and MAP4 on splits of
its own; ex_19 adds Sort & Slice and Sherlock and uses the splits of ex_14, so that the HDF results (and the HDF
column of the tables) are shared with the GNN comparison.

Fingerprints, all with 2048 positions and the fixed MLP of ex_14 ((100, 100), lr 1e-3; no KNN):

* ``morgan``      Morgan / ECFP4, radius 2, hash-folded (the established baseline)
* ``secfp``       SECFP6 = folded MHFP6, radius 3 (Probst & Reymond 2018; the MHFP default)
* ``map4``        folded MAP4, radius 2 (Capecchi et al. 2020; the MAP4 default)
* ``sort_slice``  ECFP4 vectorised via Sort & Slice (Dablander et al. 2024): the 2048 substructures that occur in
                  the most training molecules, each with its own position (``graph_hdc/baselines/sort_and_slice.py``)
* ``sherlock``    Sherlock fingerprint (Xu et al. 2026): the 2048 circular substructures of radius <= 6 with the
                  highest binary entropy over the training molecules (``graph_hdc/baselines/sherlock.py``). The
                  vocabulary is fitted on the training split of every run (SHERLOCK_DICTIONARY_PATH=None), as the
                  original ranks the substructures over the molecules it is applied to; a dictionary fitted on
                  natural products (COCONUT + LOTUS) leaves most of the 2048 positions unused on these datasets
                  (16 used positions on COMPAS-3X; user decision 2026-10-09).

The vocabularies of Sort & Slice and Sherlock are fitted without labels and without the validation and test
molecules. HDF is not run here: ``analyze_ex_19.py`` takes the HDF results (MLP) of the current ex_14 round
(prefix ex_14_gnn) and checks that every fingerprint run used the same split as the HDF run of its target and
seed. The splits are the same because the caching is left on: like ex_14, every run loads the cached,
seed-independent dataset order (``load__`` caches) and draws its split from it with the seed.

The runs therefore have to execute where the ex_14 archives and ``load__`` caches are: on KCIST in the pinned
worktree of the ex_14 re-run (``~/Programming/graph_hdc_ex14``, moved to the commit with this file; its
fingerprints/.cache is a symlink to the main checkout's), with the work-queue runner of ex_14:

    python _slurm_ex_19.py          # writes _ex19/ex19_commands.txt (all runs, longest first) and _ex19/ex19_smoke.txt
    CODE=$HOME/Programming/graph_hdc_ex14 EX14_STAGE=retry RETRY_FILE=_ex19/ex19_smoke.txt sbatch --job-name=ex19smoke --time=01:00:00 run_ex14_kcist.sbatch
    CODE=$HOME/Programming/graph_hdc_ex14 EX14_STAGE=gnn GNN_FILE=_ex19/ex19_commands.txt sbatch --job-name=ex19 --array=0-3 --time=08:00:00 run_ex14_kcist.sbatch

(the gnn stage of that runner simply shards a command file over the array tasks; its logs go to _ex14/run/).
"""
import os

from _slurm_ex_14 import DATASETS, DOWNSTREAM, _command

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex19')

PREFIX = 'ex_19_fp'
PREFIX_SMOKE = 'ex_19_smoke'
SEEDS = list(range(10))

# the MLP of ex_14 for every fingerprint; caching stays on (default) so that the cached dataset order of ex_14
# determines the splits
MLP = {
    'MODELS': ['neural_net2'],
    'NN_HIDDEN_LAYER_SIZES': DOWNSTREAM['NN_HIDDEN_LAYER_SIZES'],
    'NN_LEARNING_RATE_INIT': DOWNSTREAM['NN_LEARNING_RATE_INIT'],
}
# (variant name, experiment module suffix, parameters)
REPRESENTATIONS = [
    ('morgan', 'fp', {**MLP, 'FINGERPRINT_TYPE': 'morgan', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 2}),
    ('secfp', 'fp', {**MLP, 'FINGERPRINT_TYPE': 'secfp', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 3}),
    ('map4', 'fp', {**MLP, 'FINGERPRINT_TYPE': 'map4', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 2}),
    ('sort_slice', 'fp', {**MLP, 'FINGERPRINT_TYPE': 'sort_slice', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 2}),
    ('sherlock', 'sherlock', {**MLP, 'FINGERPRINT_SIZE': 2048, 'SHERLOCK_RADIUS': 6,
                              'SHERLOCK_DICTIONARY_PATH': None, 'SHERLOCK_FIT_JOBS': 4}),
]


def expected_cost(variant: str, ds: tuple) -> int:
    """Rough cost class of a run, to order the command file longest first (the sharding is round-robin)."""
    size = 2 if ds[1] == 'qm9_smiles' else (1 if ds[1] == 'compas_3x' else 0)
    return 2 * size + (variant in ('map4', 'sherlock'))


def build(prefix: str, datasets: list, seeds: list) -> list:
    """All command lines, longest first (stable: seed-major, dataset-minor order within a cost class)."""
    runs = [(expected_cost(variant, ds), _command(module, prefix, seed, ds, params))
            for seed in seeds for ds in datasets for variant, module, params in REPRESENTATIONS]
    return [line for _, line in sorted(runs, key=lambda r: -r[0])]


def _write(name: str, lines: list):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {len(lines):4d} commands -> _ex19/{name}')


if __name__ == '__main__':
    _write('ex19_commands.txt', build(PREFIX, DATASETS, SEEDS))
    _write('ex19_smoke.txt', build(PREFIX_SMOKE, [d for d in DATASETS if d[0] == 'freesolv_hfe'], [0]))
