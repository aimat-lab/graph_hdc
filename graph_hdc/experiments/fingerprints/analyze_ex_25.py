"""
Analysis for Experiment 25 (ablation of the global descriptors of HDF, reviewer comment R3.3). See ``_slurm_ex_25.py``.

Collects the MLP test results of the four arms (hdf, hdf_noglobal, morgan, morgan_desc) on the 17 targets of Table 1:
all arms from the ex_25 archives of the given prefix, and on the nine targets of ex_14 also hdf from the current ex_14
round (prefix ex_14_gnn) and morgan from ex_19 (prefix ex_19_fp), which ex_25 does not re-run. The arm of an archive is
determined from its parameters (``_slurm_ex_25.arm_of_params``), so that runs with other settings under the same
prefixes (e.g. the gaussian codebooks of the earlier ex_14 rounds) are never counted. If an arm of a (target, seed)
exists in ex_25 and in ex_14 / ex_19 (only in the smoke test, which re-runs the configurations of ex_14 / ex_19 to check
that they reproduce), the ex_14 / ex_19 record is used and both results are printed.

All runs of a (target, seed) must have used the same split (train, validation and test indices, in order), on which the
paired tests rely. The reference split is that of the ex_14 / ex_19 run where one exists, otherwise the most frequent
one; runs with another split are printed and excluded.

Paired contrasts per target, over the seeds that both arms have (relative change of the test MAE, (b - a) / a, mean
over the seeds with a t-interval; seeds where b is better; two-sided Wilcoxon signed-rank test; Holm-corrected over the
four contrasts of the target, the same family as in the SI tables of ex_14 / ex_19):

* ``hdf_noglobal vs hdf``         effect of removing the size and diameter encodings from HDF (reviewer's option 1)
* ``morgan_desc vs morgan``       effect of adding the size and diameter to Morgan (reviewer's option 2)
* ``morgan_desc vs hdf``          does Morgan with the descriptors reach HDF?
* ``hdf_noglobal vs morgan``      HDF's message passing alone against the plain Morgan fingerprint

Writes to ``_ex25/``:

* ``results_<prefix>.csv``                    one row per (arm, target, seed): test MAE / R2 and the source prefix
* ``contrasts_<prefix>.csv`` / ``.md``        the paired contrasts above, and the mean +- std per arm
* ``descriptor_ablation_mlp_<prefix>.tex``    SI-style table (mean, std below; best mean per target bold), with a
                                              triangle on the descriptor-free HDF column relative to HDF and on the
                                              Morgan column with descriptors relative to Morgan (filled if p < 0.05,
                                              Holm-corrected within the target as above)

Usage:
    python analyze_ex_25.py                       # prefix ex_25_desc; HDF / Morgan of ex_14_gnn / ex_19_fp
    python analyze_ex_25.py ex_25_smoke           # the KCIST smoke test
    python analyze_ex_25.py ex_25_desc --csv _ex25/results_ex_25_desc.csv   # rebuild the outputs from a results CSV
"""
import os
import csv
import json
import glob
import argparse
from collections import Counter

import numpy as np
from scipy.stats import wilcoxon, t as t_dist

from analyze_ex_14 import holm, si_cell, HARTREE_EV, ROW_META as EX14_ROW_META
from _slurm_ex_25 import TABLE1_NOTES, EX14_NOTES, MLP, arm_of_params

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex25')

HDF_PREFIX = 'ex_14_gnn'
MORGAN_PREFIX = 'ex_19_fp'
ARMS = [('hdf', 'HDF'), ('hdf_noglobal', 'HDF w/o size, diameter'), ('morgan', 'Morgan'),
        ('morgan_desc', 'Morgan + size, diameter')]
CONTRASTS = [('hdf_noglobal', 'hdf'), ('morgan_desc', 'morgan'), ('morgan_desc', 'hdf'), ('hdf_noglobal', 'morgan')]
# the triangles of the table: column -> reference column
TABLE_MARKS = {'hdf_noglobal': 'hdf', 'morgan_desc': 'morgan'}
FIELDS = ['arm', 'dataset', 'seed', 'mae', 'r2', 'source']

# Table rows in the order of Table 1: dataset name, quantity, unit and the factor that converts the raw MAE into that
# unit, as in the SI tables of ex_14 (QM9 energies are in hartree, the gaps and the ZPVE given in millihartree, the
# COMPAS-3X energies are in eV). ClogP is computed by mixin_clogp after the labels of AqSolDB have been standardized,
# so its MAE, like in Table 1, is multiplied by the standard deviation of logS in the training split.
ROW_META = {
    **EX14_ROW_META,
    'hopv15_voc': ('HOPV15', '$V_{oc}$', 'V', 1.0),
    'clogp': ('ClogP', 'ClogP', '', 1.0),
    'compas_energy': ('COMPAS-3X', '$U_0$', 'eV', 1.0),
    'compas_dipole': ('COMPAS-3X', r'$\mu$', 'D', 1.0),
    'qm9_dipole': ('QM9', r'$\mu$', 'D', 1.0),
    'qm9_cv': ('QM9', '$C_v$', r'cal/(mol\,K)', 1.0),
    'qm9_alpha': ('QM9', r'$\alpha$', '$a_0^3$', 1.0),
    'qm9_enthalpy': ('QM9', r'$\Delta H$', 'Ha', 1.0),
}
assert set(ROW_META) >= set(TABLE1_NOTES) and ROW_META['compas_gap'][3] == 1000.0 / HARTREE_EV


def archives(prefixes: tuple):
    """(start time, module, params, folder) of every completed archive of predict_molecules__hdc / __fp with a prefix."""
    out = []
    for module in ('hdc', 'fp'):
        for meta_path in glob.glob(os.path.join(RESULTS, f'predict_molecules__{module}', '*', 'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            # a killed or timed-out run keeps status 'running' (and has_error False), so require 'done'
            if params.get('__PREFIX__') not in prefixes or meta.get('status') != 'done' or meta.get('has_error'):
                continue
            out.append((meta.get('start_time') or 0, module, params, os.path.dirname(meta_path)))
    return sorted(out, key=lambda a: a[0])


def mlp_matches(params: dict) -> bool:
    """Whether the run trained the MLP of ex_14 (the HDF runs of ex_14 also trained a KNN, which is not used)."""
    hidden = params.get('NN_HIDDEN_LAYER_SIZES')
    return ('neural_net2' in (params.get('MODELS') or []) and hidden is not None
            and tuple(hidden) == tuple(MLP['NN_HIDDEN_LAYER_SIZES'])
            and params.get('NN_LEARNING_RATE_INIT') == MLP['NN_LEARNING_RATE_INIT']
            and params.get('NUM_DATA') == 1.0 and params.get('NUM_TEST') == 0.1 and params.get('NN_SEED') is None)


def collect(prefix: str):
    """
    Records {arm, dataset, seed, mae, r2, source} and their splits per (arm, dataset, seed). A newer archive of the same
    run and source replaces an older one (e.g. a retry). From ex_14 / ex_19 only hdf / morgan on the ex_14 targets.
    """
    found = {}   # (source, arm, dataset, seed) -> (record, split)
    for _, module, params, folder in archives((prefix, HDF_PREFIX, MORGAN_PREFIX)):
        source = params['__PREFIX__']
        arm = arm_of_params(module, params)
        note = params.get('NOTE')
        if arm is None or note not in TABLE1_NOTES or not mlp_matches(params):
            continue
        if source != prefix and not (note in EX14_NOTES and arm == ('hdf' if source == HDF_PREFIX else 'morgan')):
            continue
        try:
            data = json.load(open(os.path.join(folder, 'experiment_data.json')))
        except Exception as error:
            print(f'skipping unreadable archive {folder}: {error}')
            continue
        metric = data.get('metrics', {}).get('test_neural_net2')
        if metric is None:
            continue
        indices = data.get('indices', {})
        split = tuple(tuple(indices.get(part) or ()) for part in ('train', 'val', 'test'))
        record = {'arm': arm, 'dataset': note, 'seed': params['SEED'], 'mae': metric['mae'], 'r2': metric['r2'],
                  'source': source}
        found[(source, arm, note, params['SEED'])] = (record, split)

    # one record per (arm, dataset, seed): ex_14 / ex_19 first; the ex_25 duplicates (smoke test) are compared
    chosen = {}
    for (source, arm, note, seed), (record, split) in found.items():
        key = (arm, note, seed)
        if key in chosen:
            continue
        reused = [(r, s) for (src, a, n, sd), (r, s) in found.items() if (a, n, sd) == key and src != prefix]
        own = found.get((prefix, arm, note, seed))
        if reused and own:
            (ref, ref_split), (new, new_split) = reused[0], own
            print(f'REPRODUCTION {arm} {note} seed {seed}: {ref["source"]} MAE {ref["mae"]:.6g}, {prefix} MAE '
                  f'{new["mae"]:.6g} ({100 * (new["mae"] - ref["mae"]) / ref["mae"]:+.3f}%), '
                  f'split {"identical" if new_split == ref_split else "DIFFERENT"}')
        chosen[key] = reused[0] if reused else own
    return chosen


def check_splits(chosen: dict) -> set:
    """Keys (arm, dataset, seed) whose split differs from the reference split of their (dataset, seed); printed."""
    by_pair = {}
    for (arm, note, seed), (record, split) in chosen.items():
        by_pair.setdefault((note, seed), []).append((arm, record['source'], split))
    excluded = set()
    for (note, seed), runs in sorted(by_pair.items()):
        reused = [split for arm, source, split in runs if source in (HDF_PREFIX, MORGAN_PREFIX)]
        reference = reused[0] if reused else Counter(split for *_, split in runs).most_common(1)[0][0]
        for arm, source, split in runs:
            if not split[2] or split != reference:
                excluded.add((arm, note, seed))
                print(f'SPLIT MISMATCH: {arm} ({source}) {note} seed {seed}: '
                      + ('no recorded test indices' if not split[2] else 'differs from the reference split'))
    return excluded


def by_seed(records: list, arm: str, dataset: str) -> dict:
    return {r['seed']: r['mae'] for r in records if r['arm'] == arm and r['dataset'] == dataset}


def contrast(records: list, dataset: str, arm: str, reference: str) -> dict:
    """Paired comparison of ``arm`` against ``reference`` on one target (see the module docstring)."""
    a, b = by_seed(records, reference, dataset), by_seed(records, arm, dataset)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    rel = np.array([(b[s] - a[s]) / a[s] for s in seeds])
    half = t_dist.ppf(0.975, len(seeds) - 1) * rel.std(ddof=1) / np.sqrt(len(seeds))
    diffs = [b[s] - a[s] for s in seeds]
    p = wilcoxon([a[s] for s in seeds], [b[s] for s in seeds]).pvalue \
        if len(seeds) >= 5 and any(d != 0 for d in diffs) else None
    return {'dataset': dataset, 'arm': arm, 'reference': reference, 'n': len(seeds),
            'rel_change': 100 * rel.mean(), 'ci_low': 100 * (rel.mean() - half), 'ci_high': 100 * (rel.mean() + half),
            'better': sum(d < 0 for d in diffs), 'p': p}


def all_contrasts(records: list) -> list:
    rows = []
    for dataset in TABLE1_NOTES:
        family = [c for c in (contrast(records, dataset, arm, ref) for arm, ref in CONTRASTS) if c]
        for c, p_holm in zip(family, holm([c['p'] for c in family])):
            c['p_holm'] = p_holm
        rows += family
    return rows


def fmt_p(p):
    return '--' if p is None else (f'{p:.3f}' if p >= 0.001 else f'{p:.1e}')


def write_contrasts(records: list, rows: list, prefix: str):
    with open(os.path.join(OUT, f'contrasts_{prefix}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'arm', 'reference', 'n', 'rel_change', 'ci_low', 'ci_high',
                                               'better', 'p', 'p_holm'])
        writer.writeheader()
        writer.writerows(rows)

    lines = ['Test MAE of the MLP (mean +- std over the seeds, number of seeds), in the units of the table', '',
             '| Target | ' + ' | '.join(label for _, label in ARMS) + ' |', '|---|' + '---|' * len(ARMS)]
    for dataset in TABLE1_NOTES:
        name, quantity, unit, scale = ROW_META[dataset]
        cells = []
        for arm, _ in ARMS:
            values = [scale * v for v in by_seed(records, arm, dataset).values()]
            cells.append(f'{np.mean(values):.4g} +- {np.std(values):.2g} ({len(values)})' if values else '--')
        lines.append(f'| {name} {quantity}' + (f' [{unit}]' if unit else '') + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Paired change of the test MAE of the first arm relative to the second (negative = lower error); '
              'Holm over the four contrasts of a target', '',
              '| Target | Contrast | n | Change (mean, 95% CI) | first arm better in | Wilcoxon p | Holm p |',
              '|---|---|---|---|---|---|---|']
    for c in rows:
        name, quantity, *_ = ROW_META[c['dataset']]
        lines.append(f'| {name} {quantity} | {c["arm"]} vs. {c["reference"]} | {c["n"]} | {c["rel_change"]:+.1f}% '
                     f'[{c["ci_low"]:+.1f}, {c["ci_high"]:+.1f}] | {c["better"]}/{c["n"]} | {fmt_p(c["p"])} | '
                     f'{fmt_p(c["p_holm"])} |')
    content = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT, f'contrasts_{prefix}.md'), 'w') as f:
        f.write(content)
    print(content)


def write_table(records: list, rows: list, prefix: str):
    """SI-style table, one row per target in the order of Table 1 (see the module docstring)."""
    p_holm = {(c['dataset'], c['arm'], c['reference']): c['p_holm'] for c in rows}
    header = [' & & \\multicolumn{2}{c}{HDF} & \\multicolumn{2}{c}{Morgan} \\\\',
              '\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}',
              'Dataset & Quantity & Default & w/o size, diameter & Default & + size, diameter \\\\']
    lines = ['\\begin{tabular}{llcccc}', '\\toprule', *header, '\\midrule']
    previous = None
    for dataset in TABLE1_NOTES:
        name, quantity, unit, scale = ROW_META[dataset]
        cells = {arm: [scale * v for v in by_seed(records, arm, dataset).values()] for arm, _ in ARMS}
        if not all(cells.values()):
            continue
        means = {arm: np.mean(values) for arm, values in cells.items()}
        best = min(means, key=means.get)
        out = [name if name != previous else '', quantity + (f' \\qunit{{{unit}}}' if unit else '')]
        previous = name
        for arm, _ in ARMS:
            reference = TABLE_MARKS.get(arm)
            mark = significant = None
            if reference:
                mark = 'down' if means[arm] < means[reference] else 'up'
                p = p_holm.get((dataset, arm, reference))
                significant = p is not None and p < 0.05
            text = si_cell(cells[arm], bold=(arm == best), mark=mark, significant=bool(significant))
            if len(cells[arm]) < 10:
                text += f' ({len(cells[arm])})'   # flags incomplete cells (fewer seeds than planned)
            out.append(text)
        lines.append(' &\n'.join(out) + ' \\\\\n')
    lines += ['\\bottomrule', '\\end{tabular}',
              '% generated by analyze_ex_25.py; test MAE of the MLP, mean (std below) over the seeds; bold = best mean '
              'per target; triangle down/up = lower/higher MAE than the default of the same fingerprint, filled if '
              'significant (paired Wilcoxon, Holm-corrected over the four contrasts of the target, p < 0.05)']
    content = '\n'.join(lines) + '\n'
    name = f'descriptor_ablation_mlp_{prefix}.tex'
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(content)
    print(f'--- {name}\n{content}')


def main(prefix: str, csv_path: str = None):
    os.makedirs(OUT, exist_ok=True)
    if csv_path:
        with open(csv_path) as f:
            records = [{**row, 'seed': int(row['seed']), 'mae': float(row['mae']),
                        'r2': float(row['r2']) if row['r2'] else None} for row in csv.DictReader(f)]
        print(f'loaded {len(records)} records from {csv_path}')
    else:
        chosen = collect(prefix)
        excluded = check_splits(chosen)
        print(f'split check: {len(excluded)} runs without the split of their (target, seed)'
              + (' -- EXCLUDED' if excluded else ''))
        records = [record for key, (record, _) in sorted(chosen.items()) if key not in excluded]
        sources = Counter((r['source'], r['arm']) for r in records)
        print('records per source and arm: ' + ', '.join(f'{s} {a} {n}' for (s, a), n in sorted(sources.items())))
        with open(os.path.join(OUT, f'results_{prefix}.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda r: (TABLE1_NOTES.index(r['dataset']), r['arm'], r['seed'])))
    if not records:
        return
    for dataset in TABLE1_NOTES:
        missing = [label for arm, label in ARMS if not by_seed(records, arm, dataset)]
        if missing and len(missing) < len(ARMS):
            print(f'WARNING: {dataset} has no results for: {", ".join(missing)}')
    rows = all_contrasts(records)
    write_contrasts(records, rows, prefix)
    write_table(records, rows, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of ex_25 (ablation of the global descriptors of HDF).')
    parser.add_argument('prefix', nargs='?', default='ex_25_desc')
    parser.add_argument('--csv', default=None, help='rebuild the outputs from this results CSV instead of the archives')
    args = parser.parse_args()
    main(args.prefix, args.csv)
