"""
Local smoke test of ex_29 (before deploying to JUPITER): runs in the exact command format of _slurm_ex_29 on small
datasets, which exercise every code path of the three figures, a check of their archives, and analyze_ex_29.py on them.

1. Figure 4: the MLP of Table 1 on AqSolDB at 64 dimensions, HDF (depth 1, seeds 0 and 1) and Morgan (radius 1, seed
   0). The MAEs must equal those of the smoke run before the chunked encoding (HDC_ENCODE_BATCH) was introduced.
2. Figure 3b: the KNN on FreeSolv at 32 dimensions, HDF and Morgan with the native and the Euclidean distances (seed 0),
   and HDF encoded in 13 chunks of 50 molecules instead of one chunk: the same MAE.
3. SI Figure S1: the warmup (encoding cache of HOPV15, seed 0, depth 1, 2048 dimensions), then the scikit-learn MLP on
   10 and 50 training molecules, HDF (reading the cache) and Morgan, with the two HOPV15 targets in one run standing in
   for the four QM9 targets; and HDF without the cache: the same MAEs. The warmup's cache file is removed afterwards.

    python _smoke_ex_29.py         (from the fingerprints folder; prefixes ex_29_smoke_*; writes _ex27/ex29_smoke/)
"""
import os
import glob
import json
import shlex
import subprocess

import numpy as np

import _slurm_ex_29 as s29
from _slurm_ex_27 import PATH, TARGET_INDEX, _command

PREFIXES = {'size': 'ex_29_smoke_size', 'knn': 'ex_29_smoke_knn', 'train': 'ex_29_smoke_train',
            'chunk': 'ex_29_smoke_chunk', 'warm': 'ex_29_smoke_warm', 'nocache': 'ex_29_smoke_nocache'}
HOPV_NOTES = ['hopv15_pce', 'hopv15_voc']
# the smoke MAEs of 2026-10-23 before the chunked encoding (AqSolDB, 64 dimensions, depth / radius 1)
REFERENCE = {('hdf', 0): 0.6266311362184451, ('hdf', 1): 0.6586053139882391, ('morgan', 0): 1.028866675453102}


def run(line: str) -> tuple:
    """Run one command line; (archive folder, experiment data) of the new archive."""
    module = 'hdc' if 'predict_molecules__hdc.py' in line else 'fp'
    prefix = next(p for p in PREFIXES.values() if f'--__PREFIX__="\'{p}\'"' in line)
    pattern = os.path.join(PATH, 'results', f'predict_molecules__{module}', f'{prefix}__*')
    before = set(glob.glob(pattern))
    result = subprocess.run(shlex.split(line), cwd=PATH, capture_output=True, text=True)
    new = set(glob.glob(pattern)) - before
    if result.returncode != 0 or len(new) != 1:
        print(result.stderr[-3000:])
        raise SystemExit(f'FAILED: exit {result.returncode}, {len(new)} archives: {line[:200]}')
    folder = new.pop()
    return folder, json.load(open(os.path.join(folder, 'experiment_data.json')))


def params_of(folder: str) -> dict:
    meta = json.load(open(os.path.join(folder, 'experiment_meta.json')))
    return {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}


def expect(condition: bool, message: str):
    print(('ok   ' if condition else 'FAIL ') + message)
    if not condition:
        raise SystemExit(f'smoke test failed: {message}')


def test(data: dict) -> tuple:
    return tuple(sorted(data.get('indices', {}).get('test', [])))


def mae(data: dict, model: str, suffix: str = '') -> float:
    return data.get('metrics', {}).get(f'test_{model}{suffix}', {}).get('mae')


def main():
    caches_before = set(glob.glob(os.path.join(PATH, '.cache', 'hdc_*')))

    # 1. Figure 4
    print('--- Figure 4 (MLP of Table 1, AqSolDB, 64 dimensions)')
    f4 = {}
    for method, seed in (('hdf', 0), ('hdf', 1), ('morgan', 0)):
        folder, data = run(s29.size_command(PREFIXES['size'], seed, 'aqsoldb_logs', method, 64, 1))
        f4[(method, seed)] = (folder, data)
        value = mae(data, 'neural_net2')
        expect(value == REFERENCE[(method, seed)], f'{method} seed {seed}: test MAE {value} (as before the chunking)')
    params = params_of(f4[('hdf', 0)][0])
    expect(all(params.get(k) == v for k, v in {'CODEBOOK_SEED': None, 'HDC_CACHE': False, 'SPECTRUM': 'unit',
                                               'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total',
                                               'GRAPH_ATTRIBUTES': True, 'EMBEDDING_SIZE': 64, 'NUM_LAYERS': 1,
                                               'HDC_ENCODE_BATCH': s29.encode_batch(64)}.items()),
           'HDF parameters: standard encoder, CODEBOOK_SEED None, HDC_CACHE False, HDC_ENCODE_BATCH')
    expect(test(f4[('hdf', 0)][1]) == test(f4[('morgan', 0)][1]) != (),
           'HDF and Morgan share the test molecules of seed 0')
    expect(test(f4[('hdf', 0)][1]) != test(f4[('hdf', 1)][1]), 'seeds 0 and 1 have different test molecules')
    nets = [os.path.join(f4[('hdf', s)][0], 'hyper_net.pth') for s in (0, 1)]
    expect(all(os.path.exists(n) for n in nets) and open(nets[0], 'rb').read() != open(nets[1], 'rb').read(),
           'the HDF encoders (codebooks) of seeds 0 and 1 differ')

    # 2. Figure 3b
    print('--- Figure 3b (KNN, FreeSolv, 32 dimensions, both distances)')
    f3 = {}
    for variant, metrics in s29.KNN_VARIANTS.items():
        for method, metric in metrics.items():
            folder, data = run(s29.knn_command(PREFIXES['knn'], 0, 'freesolv_hfe', method, 32, metric))
            f3[(variant, method)] = data
            value = mae(data, 'k_neighbors')
            expect(value is not None and np.isfinite(value) and params_of(folder)['KN_METRIC'] == metric
                   and params_of(folder)['KN_BACKEND'] == 'torch', f'{variant} {method} ({metric}): test MAE {value}')
    expect(len({test(d) for d in f3.values()}) == 1, 'all KNN runs share the test molecules')
    ratios = {v: mae(f3[(v, 'morgan')], 'k_neighbors') / mae(f3[(v, 'hdf')], 'k_neighbors') for v in s29.KNN_VARIANTS}
    print(f'     MAE ratio Morgan / HDF at 32 dimensions (FreeSolv, seed 0): {ratios}')
    line = s29.knn_command(PREFIXES['chunk'], 0, 'freesolv_hfe', 'hdf', 32, 'cosine')
    line = line.replace(f'--HDC_ENCODE_BATCH="{s29.encode_batch(32)}"', '--HDC_ENCODE_BATCH="50"')
    folder, data = run(line)
    expect(params_of(folder)['HDC_ENCODE_BATCH'] == 50 and mae(data, 'k_neighbors') == mae(f3[('native', 'hdf')],
                                                                                         'k_neighbors'),
           f'HDF encoded in chunks of 50: the same KNN MAE {mae(data, "k_neighbors")}')

    # 3. SI Figure S1
    print('--- SI Figure S1 (warmup cache, scikit-learn MLP, HOPV15 with two targets, 10 and 50 molecules)')
    warm = _command('hdc', PREFIXES['warm'], 0, 'hopv15', HOPV_NOTES[:1],
                    {**s29.representation('hdf', s29.TRAIN_DIMENSION, 1), 'HDC_CACHE': True, 'MODELS': []})
    run(warm)
    key = f'hdc_hopv15_exp__seed_0__size_{s29.TRAIN_DIMENSION}__depth_1__bidir__totalh__unitspec'
    written = glob.glob(os.path.join(PATH, '.cache', key + '.pkl*'))
    expect(len(written) == 1, f'the warmup wrote the encoding cache {key}')
    s1 = {}
    for method in s29.METHODS_29:
        for num_train in (10, 50):
            folder, data = run(s29.train_command(PREFIXES['train'], 0, HOPV_NOTES, method, 1, num_train))
            s1[(method, num_train)] = data
            maes = [mae(data, 'neural_net', f'__t{TARGET_INDEX[note]}') for note in HOPV_NOTES]
            n_train = len(data.get('indices', {}).get('train', []))
            expect(all(m is not None and np.isfinite(m) for m in maes) and n_train == num_train
                   and params_of(folder)['TARGET_INDICES'] == [TARGET_INDEX[n] for n in HOPV_NOTES]
                   and (method != 'hdf' or params_of(folder)['HDC_CACHE'] is True),
                   f'{method} NUM_TRAIN {num_train}: {n_train} training molecules, test MAEs {maes}')
    expect(len({test(d) for d in s1.values()}) == 1, 'all S1 runs of seed 0 share the test molecules')
    _folder, data = run(s29.train_command(PREFIXES['nocache'], 0, HOPV_NOTES, 'hdf', 1, 50, cache=False))
    same = all(mae(data, 'neural_net', f'__t{TARGET_INDEX[n]}') == mae(s1[('hdf', 50)], 'neural_net',
                                                                      f'__t{TARGET_INDEX[n]}') for n in HOPV_NOTES)
    expect(same, 'HDF from the warmup cache and encoded in the run give the same MAEs')

    new_caches = set(glob.glob(os.path.join(PATH, '.cache', 'hdc_*'))) - caches_before
    expect(new_caches == set(written), f'no other HDF encoding cache written: {sorted(new_caches)}')
    for path in written:
        os.remove(path)

    # 4. the analysis on the smoke archives (output folder _ex27/ex29_smoke/)
    print('--- analyze_ex_29.py on the smoke prefixes')
    result = subprocess.run(['python', 'analyze_ex_29.py', PREFIXES['size'], PREFIXES['knn'], PREFIXES['train'],
                             'ex29_smoke'], cwd=PATH, capture_output=True, text=True)
    print(result.stdout[-2500:])
    expect(result.returncode == 0, 'analyze_ex_29.py ran')
    out = os.path.join(PATH, '_ex27', 'ex29_smoke')
    for name, n in (('fig4_runs.csv', 3), ('fig3b_runs.csv', 4), ('s1_runs.csv', 2 * 2 * len(HOPV_NOTES))):
        rows = list(csv_rows(os.path.join(out, name)))
        expect(len(rows) == n and all(row['mae'] not in ('', 'None') for row in rows),
               f'{name}: {len(rows)} rows, all with a test MAE')
    print('smoke test passed')


def csv_rows(path: str):
    import csv
    return csv.DictReader(open(path))


if __name__ == '__main__':
    main()
