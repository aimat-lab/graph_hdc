"""
Analysis of Experiment 16 (ex_16): collision rates of HDF, binary Morgan fingerprints and 1-WL.

Collects the ``molecule_collisions.py`` archives of one prefix, prints the collision rates (fraction of molecules that
share their representation with at least one other distinct molecule) per dataset, method and embedding size, the HDF
diagnostics (floating-point noise floor, collision tolerance, close but distinct pairs, collisions beyond 1-WL), and
draws the figure for the response to reviewer comment R3.2.

    python analyze_ex_16.py [prefix]      # default prefix: ex_16_collisions_unitspec

Writes ``_ex16/collisions_<prefix>.md`` and ``_ex16/collisions_<prefix>.{pdf,png}``. The figure follows the style of
the paper figures (``make_figure_ged.py``): Roboto Condensed, the method colours of the paper, bold panel letters.
"""
import os
import sys
import json
import glob
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results', 'molecule_collisions')
OUT = os.path.join(PATH, '_ex16')

DATASET_LABEL = {'qm9_smiles': 'QM9', 'zinc250k': 'ZINC250k'}
# paper palette and font size (make_figure_ged.py)
COLOR_HDF = '#4CEB99'
COLOR_MORGAN = '#4C64EB'
FONT_SIZE = 11


def dataset_stats(dataset: str) -> dict:
    """
    Number of heavy atoms of the distinct molecules of ex_16 (same filtering as molecule_collisions.py: parsable,
    single fragment, at least two heavy atoms, deduplicated by canonical SMILES without stereochemistry). Cached in
    _ex16/dataset_stats_<dataset>.json, since loading and canonicalizing a dataset takes minutes; newer archives
    also store the mean as dataset/mean_num_atoms.
    """
    path = os.path.join(OUT, f'dataset_stats_{dataset}.json')
    if os.path.exists(path):
        return json.load(open(path))
    from rdkit import Chem, RDLogger
    from chem_mat_data import load_smiles_dataset
    RDLogger.DisableLog('rdApp.*')
    sizes, seen = [], set()
    for smiles in load_smiles_dataset(dataset)['smiles']:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None or len(Chem.GetMolFrags(mol)) > 1 or mol.GetNumAtoms() < 2:
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        if canonical in seen:
            continue
        seen.add(canonical)
        sizes.append(mol.GetNumAtoms())
    sizes.sort()
    stats = {'num_molecules': len(sizes), 'mean_num_atoms': sum(sizes) / len(sizes),
             'median_num_atoms': sizes[len(sizes) // 2]}
    os.makedirs(OUT, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(stats, f)
    return stats


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


def figure(data: dict, path: str, show_wl: bool = False):
    """
    Collision rate over the embedding size, one panel per dataset, in the style of the paper figures. The 1-WL rates
    can be added as gray reference lines (the table always reports them).
    """
    plt.style.use('default')
    if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Roboto Condensed'
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['svg.fonttype'] = 'none'

    datasets = [d for d in DATASET_LABEL if d in data] + [d for d in data if d not in DATASET_LABEL]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.0), squeeze=False)
    for index, (ax, dataset) in enumerate(zip(axes[0], datasets)):
        entry = data[dataset]
        for method, color, label in (('morgan', COLOR_MORGAN, 'Morgan FP (Radius 2)'),
                                     ('hdf', COLOR_HDF, 'HDF (Depth 2)')):
            dims = sorted(entry[method])
            ax.plot(dims, [100 * entry[method][d]['collisions']['collision_rate'] for d in dims], color=color,
                    lw=1.8, marker='o', ms=6, mec='black', mew=0.8, label=label, zorder=3)
        if show_wl:
            for variant, label in (('hdf_graph', '1-WL'), ('bond_orders', '1-WL with bond orders')):
                if variant in entry['wl']:
                    ax.axhline(100 * entry['wl'][variant], color='gray', ls='--', lw=1.1, alpha=0.8, zorder=1)
                    ax.annotate(label, xy=(1.0, 100 * entry['wl'][variant]), xycoords=('axes fraction', 'data'),
                                xytext=(-4, 2), textcoords='offset points', ha='right', va='bottom',
                                fontsize=FONT_SIZE - 3, color='gray', style='italic')
        ax.set_xscale('log', base=2)
        dims = sorted(set(entry['hdf']) | set(entry['morgan']))
        ax.set_xticks(dims, [str(d) for d in dims])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        values = ([100 * r['collisions']['collision_rate'] for m in ('hdf', 'morgan') for r in entry[m].values()]
                  + ([100 * v for v in entry['wl'].values()] if show_wl else []))
        if min(values) > 0:
            ax.set_yscale('log')
            # room below the lowest line for the legend and above the highest point for the dataset annotation
            ax.set_ylim(min(values) / 3, max(values) * 3)
            # ticks at 1, 2 and 5 per decade (only at the powers of ten for wide ranges), written as plain percentages
            decades = math.log10(max(values) / min(values))
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5) if decades < 2.5 else (1,)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
            # unlabeled minor ticks at every integer multiple make the logarithmic spacing visible
            ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(2, 10)))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            # a zero rate cannot be shown on a log axis
            ax.set_yscale('symlog', linthresh=0.01)
            ax.set_ylim(0, max(values + [0.01]) * 3)
        stats = dataset_stats(dataset)
        if stats['num_molecules'] != entry['num_molecules']:
            print(f'warning: {dataset} has {stats["num_molecules"]} molecules in the size statistics but '
                  f'{entry["num_molecules"]} in the archives')
        ax.set_title(DATASET_LABEL.get(dataset, dataset), fontsize=FONT_SIZE + 1, pad=8)
        ax.annotate(f'{entry["num_molecules"]:,} molecules\nmean size {stats["mean_num_atoms"]:.1f} heavy atoms',
                    xy=(0.97, 0.96), xycoords='axes fraction', ha='right', va='top', fontsize=FONT_SIZE - 1)
        ax.set_xlabel('Embedding size', fontsize=FONT_SIZE + 1)
        ax.grid(True, which='major', ls='--', alpha=0.3)
        ax.text(-0.16, 1.04, 'abcdefgh'[index], transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight='bold',
                va='bottom', ha='left')
    axes[0][0].set_ylabel('Molecules in collisions (%) $\\downarrow$', fontsize=FONT_SIZE + 1)
    axes[0][0].legend(loc='lower right', fontsize=FONT_SIZE - 1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path + '.pdf', bbox_inches='tight')
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


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
    # ex_16_collisions: first run with implicit hydrogen counts; ex_16_collisions_totalh: total H counts;
    # ex_16_collisions_unitspec: total H counts and unit-modulus codebooks (the HDF default since 2026-10-09)
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_16_collisions_unitspec')
