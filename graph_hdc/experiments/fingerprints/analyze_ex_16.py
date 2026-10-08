"""
Analysis of Experiment 16 (ex_16): collision rates of HDF, binary Morgan fingerprints and 1-WL.

Collects the ``molecule_collisions.py`` archives of one prefix, prints the collision rates (fraction of molecules that
share their representation with at least one other distinct molecule) per dataset, method and embedding size, the HDF
diagnostics (floating-point noise floor, collision tolerance, close but distinct pairs, collisions beyond 1-WL), and
draws the figure for the response to reviewer comment R3.2.

    python analyze_ex_16.py [prefix]      # default prefix: ex_16_collisions

Writes ``_ex16/collisions_<prefix>.md`` and ``_ex16/collisions_<prefix>.{pdf,png}``.
"""
import os
import sys
import json
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results', 'molecule_collisions')
OUT = os.path.join(PATH, '_ex16')

DATASET_LABEL = {'qm9_smiles': 'QM9', 'zinc250k': 'ZINC250k'}
COLOR_HDF = '#2FB877'
COLOR_MORGAN = '#4C64EB'
COLOR_WL = '#5B6068'


def load(prefix: str) -> dict:
    """{dataset: {'wl': {variant: rate}, 'hdf': {D: record}, 'morgan': {D: record}, 'num_molecules': n}}"""
    data = {}
    archives = []
    for meta_path in glob.glob(os.path.join(RESULTS, '*', 'experiment_meta.json')):
        meta = json.load(open(meta_path))
        archives.append((meta.get('start_time') or 0, meta_path, meta))
    # oldest first, so that a rerun of the same (dataset, D) overwrites the older archive
    for _, meta_path, meta in sorted(archives, key=lambda t: t[0]):
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
            continue
        run = json.load(open(os.path.join(os.path.dirname(meta_path), 'experiment_data.json')))
        entry = data.setdefault(params['DATASET_NAME'], {'wl': {}, 'hdf': {}, 'morgan': {}})
        entry['num_molecules'] = run['dataset']['num_molecules']
        for variant, record in run.get('wl', {}).items():
            entry['wl'][variant] = record['collisions']['collision_rate']
        for method in ('hdf', 'morgan'):
            for dim, record in run.get(method, {}).items():
                entry[method][int(dim)] = record
    return data


def table(data: dict) -> str:
    lines = []
    for dataset, entry in data.items():
        dims = sorted(set(entry['hdf']) | set(entry['morgan']))
        lines.append(f'### {DATASET_LABEL.get(dataset, dataset)} ({entry["num_molecules"]} distinct molecules)\n')
        lines.append('1-WL, 2 iterations (HDF graph): {:.4%}   1-WL, 2 iterations (with bond orders): {:.4%}\n'.format(
            entry['wl'].get('hdf_graph', float('nan')), entry['wl'].get('bond_orders', float('nan'))))
        lines.append('| D | HDF | Morgan | HDF noise max | HDF tolerance | close distinct pairs (min dist.) '
                     '| molecules in HDF collisions that 1-WL separates | molecules in 1-WL collisions that HDF '
                     'separates |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for dim in dims:
            h, m = entry['hdf'].get(dim, {}), entry['morgan'].get(dim, {})
            rate = lambda r: f'{r["collisions"]["collision_rate"]:.4%}' if r else '-'
            close = (f'{h["num_close_pairs"]} ({h["close_min_distance"]:.1e})'
                     if h and h.get('close_min_distance') is not None else (f'{h["num_close_pairs"]}' if h else '-'))
            lines.append(f'| {dim} | {rate(h)} | {rate(m)} | {h.get("noise_max", float("nan")):.1e} '
                         f'| {h.get("tolerance", float("nan")):.1e} | {close} '
                         f'| {h.get("num_in_hdf_collisions_split_by_wl", "-")} '
                         f'| {h.get("num_in_wl_collisions_split_by_hdf", "-")} |')
        lines.append('')
    return '\n'.join(lines)


def figure(data: dict, path: str):
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 10})
    datasets = [d for d in DATASET_LABEL if d in data] + [d for d in data if d not in DATASET_LABEL]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.9), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        entry = data[dataset]
        for method, color, label in (('morgan', COLOR_MORGAN, 'Morgan (binary, radius 2)'),
                                     ('hdf', COLOR_HDF, 'HDF ($L = 2$)')):
            dims = sorted(entry[method])
            ax.plot(dims, [100 * entry[method][d]['collisions']['collision_rate'] for d in dims],
                    marker='o', ms=3.5, lw=1.5, color=color, label=label)
        for variant, style, label in (('hdf_graph', '--', '1-WL (2 iterations), HDF graph'),
                                      ('bond_orders', ':', '1-WL (2 iterations), with bond orders')):
            if variant in entry['wl']:
                ax.axhline(100 * entry['wl'][variant], color=COLOR_WL, ls=style, lw=1.2, label=label)
        ax.set_xscale('log', base=2)
        dims = sorted(set(entry['hdf']) | set(entry['morgan']))
        ax.set_xticks(dims, [str(d) for d in dims])
        ax.minorticks_off()
        values = ([100 * r['collisions']['collision_rate'] for m in ('hdf', 'morgan') for r in entry[m].values()]
                  + [100 * v for v in entry['wl'].values()])
        if min(values) > 0:
            ax.set_yscale('log')
            ax.set_ylim(min(values) / 2, max(values) * 3)
        else:
            # a zero rate cannot be shown on a log axis
            ax.set_yscale('symlog', linthresh=0.01)
            ax.set_ylim(0, max(values + [0.01]) * 3)
        ax.set_xlabel('embedding size $D$')
        ax.set_title(f'{DATASET_LABEL.get(dataset, dataset)} ({entry["num_molecules"]:,} molecules)'
                     .replace(',', '{,}'))
        ax.grid(alpha=0.3, lw=0.5)
    axes[0][0].set_ylabel(r'molecules in collisions (\%)')
    axes[0][-1].legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=300)


def main(prefix: str):
    data = load(prefix)
    os.makedirs(OUT, exist_ok=True)
    text = table(data)
    print(text)
    with open(os.path.join(OUT, f'collisions_{prefix}.md'), 'w') as f:
        f.write(text + '\n')
    if data:
        figure(data, os.path.join(OUT, f'collisions_{prefix}'))


if __name__ == '__main__':
    # ex_16_collisions: first run with implicit hydrogen counts; ex_16_collisions_totalh: total H counts
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_16_collisions_totalh')
