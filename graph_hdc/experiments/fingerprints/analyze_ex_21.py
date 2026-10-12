"""
Analysis of Experiment 21 (ex_21): HDF without and with the optional stereo attribute.

Collects the archives of the three parts of ``_slurm_ex_21.py`` and writes ``_ex21/summary_<prefix>.md``:

* collisions (ZINC250k, stereoisomers counted as distinct molecules): fraction of molecules whose HDF vector is
  identical to that of another molecule, and how many of them differ only in their stereochemistry;
* BACE pIC50 (ex_14 configuration, 10 seeds): test MAE / RMSE / R2 of MLP and KNN, the paired difference between
  the arms of a seed, and as a reference the difference that another MLP seed alone causes (re-seeded run without
  stereo);
* enantiomer docking ranking (Adams et al., 3 seeds): test ranking accuracy and MAE of MLP and KNN, next to the values
  that Adams et al. report for their models.

    python analyze_ex_21.py [prefix]      # default prefix: ex_21u (ex_21: first round)
"""
import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex21')
MODELS = {'neural_net2': 'MLP', 'k_neighbors': 'KNN'}

# Table 2 of Adams et al. (ICLR 2022): enantiomer ranking accuracy (%), mean and standard deviation over three trials
CHIRO_REFERENCE = [
    ('ChIRo', 72.0, 0.5),
    ('SphereNet', 68.6, 0.3),
    ('DMPNN with chiral tags', 65.9, 0.6),
    ('Tetra-DMPNN (concatenate) with chiral tags', 70.1, 0.5),
    ('2D model without stereo information (ablation, Table 3)', 49.6, 0.3),
]


def archives(module: str, prefix: str) -> list:
    """(parameters, data) of every finished archive of ``module`` with the given prefix, oldest first."""
    found = []
    for meta_path in glob.glob(os.path.join(RESULTS, module, '*', 'experiment_meta.json')):
        meta = json.load(open(meta_path))
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
            continue
        data = json.load(open(os.path.join(os.path.dirname(meta_path), 'experiment_data.json')))
        found.append((meta.get('start_time') or 0, params, data))
    return [(params, data) for _, params, data in sorted(found, key=lambda t: t[0])]


def mean_sd(values: list, scale: float = 1.0, digits: int = 3) -> str:
    if not values:
        return '-'
    values = np.asarray(values, dtype=float) * scale
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return f'{values.mean():.{digits}f} ± {sd:.{digits}f} (n={len(values)})'


def collisions(prefix: str) -> str:
    rows = {}
    for params, data in archives('molecule_collisions', f'{prefix}_collisions'):
        for dim, record in data.get('hdf', {}).items():
            rows[(params['DATASET_NAME'], int(dim), bool(params['STEREO']))] = (params, data, record)
    lines = ['## Collisions (stereoisomers counted as distinct molecules)\n',
             'Without MORGAN_CHIRALITY (round ex_21, and the plain arm of ex_21u), every pair of molecules that differ '
             'only in their stereochemistry collides for the Morgan fingerprints by construction.\n',
             '| dataset | D | HDF | molecules | differ only in stereo | molecules in HDF collisions | of which stereo-only '
             '| HDF collisions split by 1-WL | noise max | Morgan chirality | molecules in Morgan collisions '
             '| of which stereo-only |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (dataset, dim, stereo), (params, data, record) in sorted(rows.items()):
        hdf = record['collisions']
        morgan_record = data.get('morgan', {}).get(str(dim), {})
        morgan = morgan_record.get('collisions', {})
        lines.append(
            f'| {dataset} | {dim} | {"with stereo" if stereo else "without stereo"} '
            f'| {data["dataset"]["num_molecules"]} | {data["dataset"]["stereo_groups"]["num_in_collisions"]} '
            f'| {hdf["num_in_collisions"]} ({hdf["collision_rate"]:.4%}) | {record["num_in_stereo_only_collisions"]} '
            f'| {record["num_in_hdf_collisions_split_by_wl"]} | {record["noise_max"]:.1e} '
            f'| {"yes" if params.get("MORGAN_CHIRALITY") else "no"} '
            f'| {morgan.get("num_in_collisions", "-")} ({morgan.get("collision_rate", float("nan")):.4%}) '
            f'| {morgan_record.get("num_in_stereo_only_collisions", "-")} |')
    return '\n'.join(lines) + '\n'


def paired_table(runs: dict, metrics: list, arm: str = 'stereo', reference: str = 'plain') -> list:
    """runs: {(seed, arm): data}; one row per (model, metric) with both arms and their paired difference over the
    seeds that have both."""
    lines = [f'| model | metric | {reference} | {arm} | difference ({arm} - {reference}) | seeds better ({arm}) '
             f'| Wilcoxon p |', '|---|---|---|---|---|---|---|']
    seeds = sorted({seed for seed, a in runs if a == arm and (seed, reference) in runs})
    for model, label in MODELS.items():
        for metric, scale, better in metrics:
            get = lambda seed, a: runs[(seed, a)].get('metrics', {}).get(f'test_{model}', {}).get(metric)
            both = [s for s in seeds if get(s, reference) is not None and get(s, arm) is not None]
            if not both:
                continue
            first, second = [get(s, reference) for s in both], [get(s, arm) for s in both]
            diff = [b - a for a, b in zip(first, second)]
            wins = sum((d < 0) if better == 'lower' else (d > 0) for d in diff)
            # two-sided Wilcoxon signed-rank test paired by repetition (as in the GNN comparison of the SI); with
            # fewer than 6 repetitions no test can reach p < 0.05
            p = (f'{wilcoxon(second, first).pvalue:.3g}' if len(diff) >= 6 and any(d != 0 for d in diff)
                 else '-')
            lines.append(f'| {label} | {metric} | {mean_sd(first, scale)} | {mean_sd(second, scale)} '
                         f'| {mean_sd(diff, scale)} | {wins} of {len(diff)} | {p} |')
    return lines


def arm(params: dict) -> str:
    if params.get('STEREO'):
        return 'stereo'
    return 'plain' if params.get('NN_SEED') is None else 'plain_reseeded'


def morgan_arm(params: dict) -> str:
    return 'morgan_chiral' if params.get('FINGERPRINT_CHIRALITY') else 'morgan_plain'


def with_rmse(data: dict) -> dict:
    for key, values in data.get('metrics', {}).items():
        if 'mse' in values:
            values['rmse'] = float(np.sqrt(values['mse']))
    return data


def runs_by(stem: str, prefix: str, key=lambda params: None) -> dict:
    """{key: {(seed, arm): data}} with the HDF runs (module <stem>__hdc) and the Morgan runs (<stem>__fp)."""
    grouped = defaultdict(dict)
    for module, arm_of in ((f'{stem}__hdc', arm), (f'{stem}__fp', morgan_arm)):
        for params, data in archives(module, prefix):
            grouped[key(params)][(params['SEED'], arm_of(params))] = with_rmse(data)
    return grouped


# comparisons with the Morgan fingerprints (only shown if both arms have runs)
MORGAN_COMPARISONS = [
    ('morgan_chiral', 'morgan_plain', 'Morgan fingerprints with vs. without chirality'),
    ('stereo', 'morgan_chiral', 'HDF with stereo vs. Morgan fingerprints with chirality'),
]


def morgan_tables(runs: dict, metrics: list) -> list:
    lines = []
    for arm_, reference, title in MORGAN_COMPARISONS:
        if any(a == arm_ for _, a in runs) and any(a == reference for _, a in runs):
            lines += [f'\n{title}:\n'] + paired_table(runs, metrics, arm=arm_, reference=reference)
    return lines


def prediction_section(title: str, runs: dict) -> str:
    """Paired comparison with and without stereo, and the re-seeded reference, for one dataset."""
    metrics = [('mae', 1.0, 'lower'), ('rmse', 1.0, 'lower'), ('r2', 1.0, 'higher')]
    lines = [f'## {title} (ex_14 configuration: D = 2048, L = 2)\n']
    lines += paired_table(runs, metrics)
    lines += ['\nReference: the same representation without stereo, trained with another network seed (NN_SEED = '
              '1000 + seed). Only the MLP depends on the network seed; KNN rows are identical by construction.\n']
    lines += paired_table(runs, metrics, arm='plain_reseeded')
    lines += morgan_tables(runs, metrics)
    return '\n'.join(lines) + '\n'


def bace(prefix: str) -> str:
    return prediction_section('BACE pIC50', runs_by('predict_molecules', f'{prefix}_bace')[None])


def props(prefix: str) -> str:
    """Further datasets of the main comparison with the BACE design, one section per dataset (NOTE)."""
    by_note = runs_by('predict_molecules', f'{prefix}_props', key=lambda params: params['NOTE'])
    return '\n'.join(prediction_section(note, runs) for note, runs in sorted(by_note.items()))


def chiro(prefix: str) -> str:
    runs = runs_by('predict_enantiomers', f'{prefix}_chiro')[None]
    metrics = [('ranking_accuracy', 100.0, 'higher'), ('mae', 1.0, 'lower')]
    lines = ['## Enantiomer docking ranking (Adams et al., original splits)\n']
    lines += paired_table(runs, metrics)
    lines += morgan_tables(runs, metrics)
    ties = defaultdict(list)
    for (seed, a), data in runs.items():
        for model in MODELS:
            record = data.get('metrics', {}).get(f'test_{model}', {})
            if 'num_pairs' in record:
                ties[(model, a)].append(f'{record["num_tied_pairs"]}/{record["num_pairs"]} '
                                        f'({record.get("num_identical_pairs", "?")} identical)')
    lines.append('\nTied test pairs (predictions closer than TIE_TOLERANCE; in parentheses the pairs whose two '
                 'enantiomers have identical representations): '
                 + '; '.join(f'{MODELS[m]} {a}: {", ".join(v)}' for (m, a), v in sorted(ties.items())))
    lines.append('\nReference (Adams et al., Table 2 and 3), ranking accuracy in %. Their protocol differs from ours: '
                 'trained graph neural networks, the epoch selected by the validation ranking accuracy, both '
                 'enantiomers of a pair in every minibatch, and predictions averaged over the conformers.\n')
    lines += [f'- {name}: {value:.1f} ± {sd:.1f}' for name, value, sd in CHIRO_REFERENCE]
    return '\n'.join(lines) + '\n'


# Table 3 of Baimacheva et al. (J Cheminform 2025): per-molecule test accuracy of the elution order classification on
# their split (random forest, one model), and the mean over five further random splits. The descriptors of a molecule
# alone correspond to our "accuracy"; the variants that use the difference to the descriptors of the opposite
# enantiomer decide each molecule against its mirror image, like our "ranking_accuracy".
CMRT_REFERENCE = [
    ('chirality-aware Morgan fingerprint (radius 3, 512 bits)', 0.807),
    ('Transformer heteroencoder latent vector', 0.729),
    ('CDDD latent vector', 0.683),
    ('chirality-aware Morgan fingerprint, difference to the opposite enantiomer', 0.822),
    ('Transformer latent vector, difference to the opposite enantiomer', 0.753),
    ('CDDD latent vector, difference to the opposite enantiomer', 0.732),
    ('chirality-aware Morgan fingerprint, mean over five further random splits', 0.81),
]


def r_first_rule(params: dict, data: dict) -> float:
    """Accuracy of the rule "the R enantiomer elutes first" on the test molecules of a run. A model that receives the
    R/S descriptors can reach it without learning anything else, so it is the chance level for HDF with stereo."""
    table = pd.read_csv(params['PAIRS_CSV'])
    test = table.iloc[data['indices']['test']]
    return float(((test['r_s'] == 'R').astype(int) == test['first_eluted']).mean())


def cmrt(prefix: str) -> str:
    """Elution order on Chiralpak AD-H, one table per split mode (the authors' split, random splits by pair)."""
    by_split = runs_by('predict_enantiomers', f'{prefix}_cmrt', key=lambda params: params['SPLIT_MODE'])
    rule = defaultdict(dict)
    for p, d in archives('predict_enantiomers__hdc', f'{prefix}_cmrt'):
        rule[p['SPLIT_MODE']][p['SEED']] = r_first_rule(p, d)
    lines = ['## Elution order on Chiralpak AD-H (CMRT subset of Baimacheva et al.)\n',
             'ranking_accuracy: fraction of test pairs whose predictions are in the elution order (ties count half); '
             'accuracy: fraction of test molecules whose prediction, thresholded at 0.5, matches first/last eluted '
             '(the metric of Baimacheva et al.).\n']
    for split_mode, runs in sorted(by_split.items()):
        lines.append(f'\n### Split: {split_mode}\n')
        metrics = [('ranking_accuracy', 100.0, 'higher'), ('accuracy', 100.0, 'higher')]
        lines += paired_table(runs, metrics)
        lines += morgan_tables(runs, metrics)
        identical = defaultdict(list)
        for (seed, a), data in sorted(runs.items()):
            record = data.get('metrics', {}).get('test_neural_net2', {})
            if 'num_identical_pairs' in record:
                identical[a].append(record['num_identical_pairs'])
        if identical:
            lines.append('\nTest pairs whose two enantiomers have identical representations (per seed): '
                         + '; '.join(f'{a}: {values}' for a, values in sorted(identical.items())))
        lines.append(f'\nRule "the R enantiomer elutes first" on the same test molecules: '
                     f'{mean_sd(list(rule[split_mode].values()), 100.0, 1)}')
    lines.append('\nReference (Baimacheva et al., Table 3), per-molecule test accuracy of a random forest:\n')
    lines += [f'- {name}: {value:.3f}' for name, value in CMRT_REFERENCE]
    return '\n'.join(lines) + '\n'


def main(prefix: str):
    text = '\n'.join([f'# ex_21: HDF without and with stereo descriptors (prefix {prefix})\n',
                      collisions(prefix), bace(prefix), props(prefix), chiro(prefix), cmrt(prefix)])
    print(text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'summary_{prefix}.md'), 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_21u')
