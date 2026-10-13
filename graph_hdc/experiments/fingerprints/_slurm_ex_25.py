"""
Command-file generator for Experiment 25 (ex_25): ablation of the global descriptors of HDF (reviewer comment R3.3).

R3.3 suggests "an ablation without these global descriptors [the graph size and the graph diameter], or a comparison in
which the baseline fingerprints are augmented with the same descriptors", to isolate the contribution of the
hyperdimensional message passing. ex_25 does both (user decision 2026-10-12), on all 17 targets of Table 1, with the
fixed MLP of ex_14 ((100, 100), lr 1e-3; no KNN), seeds 0-9 and the splits of ex_14:

* ``hdf``           HDF with the standard encoder: D=2048, L=2, bidirectional message passing, total hydrogen counts,
                    unit-modulus codebooks (the HDF arm of the current ex_14 round)
* ``hdf_noglobal``  the same without the encodings of the graph size and diameter (GRAPH_ATTRIBUTES=False, ex_22:
                    the message-passing readout alone, normalized to unit length)
* ``morgan``        Morgan, radius 2, 2048 bits (the Morgan arm of ex_19)
* ``morgan_desc``   the same with the graph size and diameter appended as two features, each divided by its maximum
                    over the dataset (GRAPH_DESCRIPTORS=True)

HDF and Morgan already ran on the nine targets of ex_14 with exactly these settings and splits (HDF: ex_14 round 4,
prefix ex_14_gnn; Morgan: ex_19, prefix ex_19_fp). ``analyze_ex_25.py`` takes those records, so that this file only
writes the remaining runs: hdf_noglobal and morgan_desc on all 17 targets, hdf and morgan on the 8 targets that ex_14
leaves out (HOPV15 Voc, ClogP, COMPAS-3X U0 and dipole, QM9 dipole, alpha, enthalpy and Cv). The splits are the same
because the caching is left on: every run loads the cached, seed-independent dataset order (``load__`` caches) and
draws its split from it with the seed. The runs therefore execute where the ex_14 archives and caches are: on KCIST in
the pinned worktree of ex_14 (``~/Programming/graph_hdc_ex14``, moved to the commit with this file; its
fingerprints/.cache is a symlink to the main checkout's), with the work-queue runner of ex_14.

HDF encodings are cached per (DATASET_NAME, SEED, encoder settings), so the HDF runs of targets that share a dataset
would race on a missing cache file. The HDF runs are therefore grouped by (DATASET_NAME, SEED) into shards, and no two
shards share a cache key, so that the shards can run on different nodes at the same time. Within a shard, the first
run of every key whose cache file does not exist yet forms the primer block, which the runner executes to completion
before the rest; the runs whose cache exists (the HDF caches of ex_14) go directly into the rest. The Morgan runs share
no caches besides ``load__`` and go into one file that the gnn stage of the runner shards round-robin.

Files written to ``_ex25/``:

* ``hdf_<k>.txt`` / ``hdf_<k>_primer_count.txt``  HDF shard k (k = 0 .. HDF_SHARDS-1) and its number of primer lines
* ``fp.txt``          all Morgan runs, longest first (for the round-robin sharding of the gnn stage)
* ``smoke.txt``       the KCIST smoke test (prefix ex_25_smoke): all four arms on FreeSolv, whose HDF and Morgan runs
                      exist in ex_14 / ex_19 (the analysis checks the splits against them and compares the results of
                      the identical configurations), and on HOPV15 Voc, a target that ex_14 leaves out; seed 0. Every
                      HDF line is the only run of its cache key, so the file can run as one queue.
* ``missing_hdf_<k>.txt`` / ``missing_hdf_<k>_primer_count.txt`` / ``missing_fp.txt``  written by the missing mode:
                      the lines without a completed archive, for resuming (see write_missing)

On KCIST, from the fingerprints folder of the pinned worktree (pass the variables through the environment, see
run_ex14_kcist.sbatch; the command files must be generated there, where the caches are):

    python _slurm_ex_25.py
    CODE=$HOME/Programming/graph_hdc_ex14 EX14_STAGE=retry RETRY_FILE=_ex25/smoke.txt sbatch --job-name=ex25smoke --time=01:00:00 run_ex14_kcist.sbatch
    python analyze_ex_25.py ex_25_smoke
    python _slurm_ex_25.py            # again after the smoke: its descriptor-free caches move those lines to the rest
    CODE=... EX14_STAGE=hdf HDF_FILE=_ex25/hdf_0.txt HDF_PRIMER_FILE=_ex25/hdf_0_primer_count.txt sbatch --job-name=ex25hdf0 run_ex14_kcist.sbatch
    (the same for every HDF shard)
    CODE=... EX14_STAGE=gnn GNN_FILE=_ex25/fp.txt sbatch --job-name=ex25fp --array=0-1 run_ex14_kcist.sbatch

Resuming (no ex_25 job may be running): ``python _slurm_ex_25.py missing``, then the same submissions with
HDF_FILE=_ex25/missing_hdf_<k>.txt HDF_PRIMER_FILE=_ex25/missing_hdf_<k>_primer_count.txt and GNN_FILE=_ex25/missing_fp.txt.

Usage:
    python _slurm_ex_25.py            # writes all command files
    python _slurm_ex_25.py --check    # only prints the run counts per arm and target
    python _slurm_ex_25.py missing    # writes the missing_* files
"""
import os
import re
import sys
import glob
import json

from _slurm_ex_13 import DATASETS as TABLE1_DATASETS, target_matches
from _slurm_ex_14 import (DATASETS as EX14_DATASETS, VARIANTS, HDF_ROUND, DOWNSTREAM, _command, params_of_line,
                          hdf_cache_name)
from _slurm_ex_19 import REPRESENTATIONS as EX19_REPRESENTATIONS

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex25')

PREFIX = 'ex_25_desc'
PREFIX_SMOKE = 'ex_25_smoke'
SEEDS = list(range(10))
# number of HDF command files that may run side by side on different nodes
HDF_SHARDS = 2

# The 17 targets of Table 1 (the ex_13 targets without the HOPV15 gap and Jsc, which the table does not report), in
# the tuple format of _slurm_ex_13 / _slurm_ex_14: (note, DATASET_NAME, TARGET_INDEX, extra_params, big)
TABLE1_NOTES = [
    'hopv15_pce', 'hopv15_voc', 'freesolv_hfe', 'bace_ic50', 'lipophilicity_logD', 'aqsoldb_logs', 'clogp',
    'compas_energy', 'compas_gap', 'compas_dipole',
    'qm9_dipole', 'qm9_cv', 'qm9_zpve', 'qm9_alpha', 'qm9_enthalpy', 'qm9_energy', 'qm9_gap',
]
DATASETS = [next(ds for ds in TABLE1_DATASETS if ds[0] == note) for note in TABLE1_NOTES]
# the targets of ex_14, for which the hdf and morgan arms are taken from ex_14 / ex_19
EX14_NOTES = {ds[0] for ds in EX14_DATASETS}
# ex_14 says it uses the ex_13 tuples; the reuse of its runs relies on that
assert all(ds in TABLE1_DATASETS for ds in EX14_DATASETS) and EX14_NOTES <= set(TABLE1_NOTES)

# the MLP of ex_14 without the KNN of its HDF arm (the MLP is trained first and seeds itself, so its result does not
# depend on whether a KNN follows)
MLP = {
    'MODELS': ['neural_net2'],
    'NN_HIDDEN_LAYER_SIZES': DOWNSTREAM['NN_HIDDEN_LAYER_SIZES'],
    'NN_LEARNING_RATE_INIT': DOWNSTREAM['NN_LEARNING_RATE_INIT'],
}
# the HDF encoder of the current ex_14 round (VARIANTS['hdf'] + HDF_ROUND; _command adds the bidirectional message
# passing and the total hydrogen counts), with the MLP only
HDF = {**{k: v for k, v in VARIANTS['hdf'][1].items() if k not in DOWNSTREAM}, **HDF_ROUND, **MLP}
# the Morgan fingerprint of ex_19
MORGAN = next(params for variant, module, params in EX19_REPRESENTATIONS if variant == 'morgan')
assert MORGAN['MODELS'] == MLP['MODELS'] and MORGAN['NN_HIDDEN_LAYER_SIZES'] == MLP['NN_HIDDEN_LAYER_SIZES']

# (arm, module suffix, parameters, whether the arm is taken from ex_14 / ex_19 on the ex_14 targets)
ARMS = [
    ('hdf', 'hdc', HDF, True),
    ('hdf_noglobal', 'hdc', {**HDF, 'GRAPH_ATTRIBUTES': False}, False),
    ('morgan', 'fp', MORGAN, True),
    ('morgan_desc', 'fp', {**MORGAN, 'GRAPH_DESCRIPTORS': True}, False),
]


def arm_of_params(module: str, params: dict):
    """The arm of a run of predict_molecules__<module> with these parameters, or None if it is none of the arms."""
    if module == 'hdc':
        encoder = {k: params.get(k) for k in ('EMBEDDING_SIZE', 'NUM_LAYERS', 'ENCODING_MODE', 'SPECTRUM')}
        if encoder != {k: HDF[k] for k in encoder} or params.get('BIDIRECTIONAL') is not True \
                or params.get('HYDROGEN_COUNT') != 'total':
            return None
        return 'hdf' if params.get('GRAPH_ATTRIBUTES', True) else 'hdf_noglobal'
    if module == 'fp':
        fingerprint = {k: params.get(k) for k in ('FINGERPRINT_TYPE', 'FINGERPRINT_SIZE', 'FINGERPRINT_RADIUS')}
        if fingerprint != {k: MORGAN[k] for k in fingerprint}:
            return None
        return 'morgan_desc' if params.get('GRAPH_DESCRIPTORS', False) else 'morgan'
    return None


def expected_cost(ds: tuple) -> int:
    """Rough cost class of a run (higher = longer): QM9 > COMPAS-3X > the rest, as in ex_14."""
    return 2 if ds[1] == 'qm9_smiles' else (1 if ds[1] == 'compas_3x' else 0)


def runs(prefix: str, datasets: list, seeds: list, reuse: bool = True) -> list:
    """(arm, module, ds, seed, command line) of every run; with ``reuse``, without those taken from ex_14 / ex_19."""
    out = []
    for seed in seeds:
        for ds in datasets:
            for arm, module, params, reused in ARMS:
                if reuse and reused and ds[0] in EX14_NOTES:
                    continue
                line = _command(module, prefix, seed, ds, params)
                # the analysis recognizes the arms by their parameters; every generated line must map to its arm
                assert arm_of_params(module, params_of_line(line)) == arm, line
                out.append((arm, module, ds, seed, line))
    return out


def hdf_cache_key(line: str) -> tuple:
    """The encoding cache of an HDF run: one per (DATASET_NAME, SEED, encoder settings)."""
    params = params_of_line(line)
    return params['DATASET_NAME'], params['SEED'], params.get('GRAPH_ATTRIBUTES', True)


def cache_files(line: str) -> list:
    """The existing encoding cache file(s) of an HDF run in the (linked) .cache folder."""
    return glob.glob(os.path.join(PATH, '.cache', hdf_cache_name(params_of_line(line)) + '.pkl*'))


def split_primer(lines: list) -> tuple:
    """
    (primer, rest) of HDF lines: the first line of every cache key whose cache file does not exist yet goes into the
    primer, which the runner executes to completion before the rest; every other line (later lines of such a key, and
    all lines whose cache exists, e.g. the HDF caches of ex_14) goes into the rest. Longest runs first in both.
    """
    primer, rest, primed = [], [], set()
    for line in sorted(lines, key=lambda l: -expected_cost(dataset_of_line(l))):
        key = hdf_cache_key(line)
        if key in primed or cache_files(line):
            rest.append(line)
        else:
            primer.append(line)
            primed.add(key)
    return primer, rest


def dataset_of_line(line: str) -> tuple:
    note = params_of_line(line)['NOTE']
    return next(ds for ds in DATASETS if ds[0] == note)


def hdf_shards(hdf_runs: list, count: int) -> list:
    """
    Split the HDF runs into ``count`` shards that share no (DATASET_NAME, SEED), each as (primer lines, rest lines),
    see split_primer. The (DATASET_NAME, SEED) groups are assigned greedily, largest cost first, to the currently
    cheapest shard.
    """
    groups = {}
    for arm, module, ds, seed, line in hdf_runs:
        groups.setdefault((ds[1], seed), []).append((expected_cost(ds), line))
    loads = [0] * count
    shards = [[] for _ in range(count)]
    for key, members in sorted(groups.items(), key=lambda kv: (-sum(3 ** c for c, _ in kv[1]), kv[0])):
        k = loads.index(min(loads))
        shards[k] += [line for _, line in members]
        loads[k] += sum(3 ** c for c, _ in members)
    return [split_primer(lines) for lines in shards]


def identity_of_line(line: str) -> tuple:
    """(arm, NOTE, SEED) of a command line."""
    module = re.search(r'predict_molecules__(\w+)\.py', line).group(1)
    params = params_of_line(line)
    return arm_of_params(module, params), params['NOTE'], params['SEED']


def labels_valid(note: str, data: dict) -> bool:
    """
    Whether the archive data of a run of this target come from the corrected label pipeline where that matters: the
    ClogP labels of mixin_clogp are standardized only since 2026-10-13 ("labels/scaled_after_hooks", see
    predict_molecules.py); before, the scaler was fitted on the logS labels of AqSolDB.
    """
    uses_mixin = any(ds[0] == note and ds[3].get('__INCLUDE__') == 'mixin_clogp.py' for ds in DATASETS)
    return not uses_mixin or bool(data.get('labels', {}).get('scaled_after_hooks'))


def completed(prefix: str) -> set:
    """(arm, NOTE, SEED) of every completed (status 'done', no error) archive of the prefix with the current target."""
    done = set()
    for module in ('hdc', 'fp'):
        for meta_path in glob.glob(os.path.join(PATH, 'results', f'predict_molecules__{module}', '*',
                                                'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error') \
                    or not target_matches(params):
                continue
            if params.get('NOTE') == 'clogp':
                try:
                    data = json.load(open(meta_path.replace('experiment_meta.json', 'experiment_data.json')))
                except Exception:
                    continue
                if not labels_valid('clogp', data):
                    continue
            done.add((arm_of_params(module, params), params.get('NOTE'), params.get('SEED')))
    return done


def write_missing():
    """
    Write the lines of the command files without a completed archive to missing_hdf_<k>.txt (with a new primer
    count, see split_primer) and missing_fp.txt, e.g. after a failed primer, which makes the runner skip the rest of
    its shard. The cache of a descriptor-free (__noglobal) key whose primer run did not complete may be truncated
    (pycomex compresses in place) and is deleted, so that the new primer rebuilds it; the caches of ex_14 are never
    touched. Only call this while no ex_25 job is running.
    """
    done = completed(PREFIX)
    for k in range(HDF_SHARDS):
        lines = open(os.path.join(OUT, f'hdf_{k}.txt')).read().splitlines()
        primer_count = int(open(os.path.join(OUT, f'hdf_{k}_primer_count.txt')).read())
        for line in lines[:primer_count]:
            if identity_of_line(line) not in done and params_of_line(line).get('GRAPH_ATTRIBUTES') is False:
                for path in cache_files(line):
                    print(f'removing possibly incomplete cache {os.path.basename(path)}')
                    os.remove(path)
        primer, rest = split_primer([line for line in lines if identity_of_line(line) not in done])
        _write(f'missing_hdf_{k}.txt', primer + rest)
        _write(f'missing_hdf_{k}_primer_count.txt', [str(len(primer))])
    lines = open(os.path.join(OUT, 'fp.txt')).read().splitlines()
    _write('missing_fp.txt', [line for line in lines if identity_of_line(line) not in done])


def _write(name: str, lines: list):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'wrote {len(lines):4d} lines -> _ex25/{name}')


def summary(all_runs: list):
    counts = {}
    for arm, module, ds, seed, line in all_runs:
        counts.setdefault(ds[0], {}).setdefault(arm, 0)
        counts[ds[0]][arm] += 1
    arms = [arm for arm, *_ in ARMS]
    print('runs per target and arm (taken from ex_14 / ex_19 where 0):')
    print(f'  {"target":20s} ' + ' '.join(f'{a:>13s}' for a in arms))
    for note in TABLE1_NOTES:
        print(f'  {note:20s} ' + ' '.join(f'{counts.get(note, {}).get(a, 0):13d}' for a in arms))
    print(f'  total: {len(all_runs)} runs')


if __name__ == '__main__':
    if 'missing' in sys.argv[1:]:
        write_missing()
        sys.exit(0)
    production = runs(PREFIX, DATASETS, SEEDS)
    summary(production)
    if '--check' in sys.argv[1:]:
        sys.exit(0)

    hdf_runs = [r for r in production if r[1] == 'hdc']
    fp_runs = [r for r in production if r[1] == 'fp']
    for k, (primer, rest) in enumerate(hdf_shards(hdf_runs, HDF_SHARDS)):
        _write(f'hdf_{k}.txt', primer + rest)
        _write(f'hdf_{k}_primer_count.txt', [str(len(primer))])
    # longest first (stable within a cost class), so that the round-robin shards get the same mix
    _write('fp.txt', [line for *_, line in sorted(fp_runs, key=lambda r: -expected_cost(r[2]))])

    # smoke: all four arms (no reuse) on FreeSolv and HOPV15 Voc, seed 0. Every HDF line is the only run of its
    # cache key, so the file can run as one queue. It builds the descriptor-free caches of (freesolv, 0) and
    # (hopv15_exp, 0), which the production files then use: run the smoke before the production jobs, never alongside.
    smoke = runs(PREFIX_SMOKE, [ds for ds in DATASETS if ds[0] in ('freesolv_hfe', 'hopv15_voc')], [0], reuse=False)
    smoke_hdf = [line for arm, module, ds, seed, line in smoke if module == 'hdc']
    assert len({hdf_cache_key(line) for line in smoke_hdf}) == len(smoke_hdf)
    _write('smoke.txt', smoke_hdf + [line for arm, module, ds, seed, line in smoke if module == 'fp'])
