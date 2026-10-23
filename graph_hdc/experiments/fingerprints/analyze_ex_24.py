"""
Analysis of Experiment 24 (ex_24, BO benchmark, ``bo_benchmark.py``).

Reads ``_ex24/campaigns.tsv`` (``python bo_benchmark.py collect``) and writes to ``_ex24/analysis/``:

- ``summary.tsv`` / ``summary.md``: per objective, method and size the median [first-third quartile] of the
  normalized-regret AUC over the 25 campaigns, the mean final regret, the hit rate (normalized regret <= 0.05)
  and the mean seconds per campaign.
- ``tests.tsv`` / ``tests.md``: paired two-sided Wilcoxon signed-rank tests (campaign k against campaign k, same
  initial molecules) of HDF + Matern against Morgan + Tanimoto and against Morgan-PCA + Matern on the AUC, per
  objective and size, Holm-corrected within each objective (2 comparisons x 8 sizes). HDF + linear kernel against
  both Morgan methods is a second family of the same form, and the comparisons of every method with random search
  are reported in the same table as a third family per objective.
- ``auc_vs_size.png`` / ``.pdf``: quick-look figure (median AUC with interquartile band against the size, random
  search as a dashed line), one panel per objective. The layout of the paper figure is decided once the results
  are in.

    python analyze_ex_24.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EX24_ROOT', os.path.join(PATH, '_ex24'))
OUT = os.path.join(ROOT, 'analysis')

OBJECTIVES = ['clogp', 'qed', 'solubility', 'lipophilicity', 'freesolv', 'beet', 'qm9_gap', 'qm9_dipole',
              'melting_point']
TITLES = {'clogp': 'ClogP (AqSolDB)', 'qed': 'QED (ZINC250k)', 'solubility': 'Solubility (AqSolDB)',
          'lipophilicity': 'Lipophilicity', 'freesolv': 'FreeSolv', 'beet': 'BEET', 'qm9_gap': 'HOMO-LUMO gap (QM9)',
          'qm9_dipole': 'Dipole moment (QM9)', 'melting_point': 'Melting point'}
METHODS = ['hdf_matern', 'morgan_tanimoto', 'morgan_pca', 'hdf_linear', 'random']
LABELS = {'hdf_matern': 'HDF (Matérn)', 'morgan_tanimoto': 'Morgan (Tanimoto)',
          'morgan_pca': 'Morgan-2048 PCA (Matérn)', 'hdf_linear': 'HDF (linear)', 'random': 'Random search'}
COLORS = {'hdf_matern': '#4CEB99', 'morgan_tanimoto': '#4C64EB', 'morgan_pca': '#7C92FF', 'hdf_linear': '#69F7AE',
          'random': 'gray'}
# families of HDF against both Morgan methods, each Holm-corrected within the objective. The linear kernel has its
# own family (user, 2026-10-23: the main-text figure shows HDF + linear kernel), so the Matern tests are unchanged.
COMPARISONS = {
    'hdf_vs_morgan': [('hdf_matern', 'morgan_tanimoto'), ('hdf_matern', 'morgan_pca')],
    'hdf_linear_vs_morgan': [('hdf_linear', 'morgan_tanimoto'), ('hdf_linear', 'morgan_pca')],
}


def holm(pvalues: list) -> list:
    """Holm-Bonferroni adjusted p-values (None entries are passed through), as in analyze_ex_14.py."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    adjusted = list(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired(df: pd.DataFrame, objective: str, method_a: str, size_a, method_b: str, size_b):
    """AUCs of two cells matched by campaign (only campaigns finished in both), or None."""
    def cell(method, size):
        sel = (df.objective == objective) & (df.method == method)
        sel &= df['size'].isna() if size is None else (df['size'] == size)
        return df[sel].set_index('campaign')['auc']
    a, b = cell(method_a, size_a), cell(method_b, size_b)
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return None
    return a.loc[common].to_numpy(), b.loc[common].to_numpy()


def wilcoxon_p(a: np.ndarray, b: np.ndarray):
    """Two-sided Wilcoxon signed-rank p-value; None if all paired differences are zero."""
    if np.allclose(a, b):
        return None
    return float(wilcoxon(a, b).pvalue)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (objective, method, size), group in df.groupby(['objective', 'method', 'size'], dropna=False):
        auc = group['auc'].to_numpy()
        rows.append({
            'objective': objective, 'method': method, 'size': None if pd.isna(size) else int(size),
            'campaigns': len(group),
            'auc_median': np.median(auc), 'auc_q1': np.percentile(auc, 25), 'auc_q3': np.percentile(auc, 75),
            'auc_mean': auc.mean(), 'final_regret_mean': group['final_regret'].mean(),
            'hit_rate': 100 * group['hit'].mean(), 'seconds_mean': group['seconds_total'].mean(),
            'fit_failures': int(group['fit_failures'].sum()),
        })
    return pd.DataFrame(rows)


def tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective in OBJECTIVES:
        sizes = sorted(df[(df.objective == objective) & df['size'].notna()]['size'].unique())
        # families 1 and 2: HDF + Matern / HDF + linear against both Morgan methods, all sizes, Holm within the
        # objective
        families = []
        for name, comparisons in COMPARISONS.items():
            family = []
            for size in sizes:
                for method_a, method_b in comparisons:
                    pair = paired(df, objective, method_a, size, method_b, size)
                    if pair is None:
                        continue
                    a, b = pair
                    family.append({'objective': objective, 'family': name, 'size': int(size),
                                   'method_a': method_a, 'method_b': method_b, 'campaigns': len(a),
                                   'auc_median_a': np.median(a), 'auc_median_b': np.median(b),
                                   'a_better': int((a < b).sum()), 'p': wilcoxon_p(a, b)})
            families.append(family)
        # family 3: every method against random search
        family_random = []
        for method in METHODS[:-1]:
            for size in sizes:
                pair = paired(df, objective, method, size, 'random', None)
                if pair is None:
                    continue
                a, b = pair
                family_random.append({'objective': objective, 'family': 'vs_random', 'size': int(size),
                                      'method_a': method, 'method_b': 'random', 'campaigns': len(a),
                                      'auc_median_a': np.median(a), 'auc_median_b': np.median(b),
                                      'a_better': int((a < b).sum()), 'p': wilcoxon_p(a, b)})
        for rows_family in families + [family_random]:
            for row, p_holm in zip(rows_family, holm([r['p'] for r in rows_family])):
                row['p_holm'] = p_holm
                rows.append(row)
    return pd.DataFrame(rows)


def figure(summary: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.style.use('default')
    if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Roboto Condensed'
    font_size = 11
    plt.rcParams.update({'font.size': font_size, 'svg.fonttype': 'none'})

    objectives = [o for o in OBJECTIVES if o in set(summary.objective)]
    columns = 3
    rows = int(np.ceil(len(objectives) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)
    for ax, objective in zip(axes.flat, objectives):
        data = summary[summary.objective == objective]
        for method in METHODS[:-1]:
            cell = data[data.method == method].sort_values('size')
            if cell.empty:
                continue
            ax.fill_between(cell['size'], cell['auc_q1'], cell['auc_q3'], color=COLORS[method], alpha=0.2, lw=0)
            ax.plot(cell['size'], cell['auc_median'], color=COLORS[method], lw=1.8, marker='o', ms=6, mec='black',
                    mew=0.8, label=LABELS[method], ls='--' if method == 'hdf_linear' else '-')
        random = data[data.method == 'random']
        if not random.empty:
            ax.axhline(float(random['auc_median'].iloc[0]), color='gray', ls='--', lw=1.2)
            ax.text(0.98, float(random['auc_median'].iloc[0]), 'random search', color='gray', style='italic',
                    fontsize=font_size - 2, ha='right', va='bottom', transform=ax.get_yaxis_transform())
        ax.set_xscale('log', base=2)
        sizes = sorted(data['size'].dropna().unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(int(s)) for s in sizes])
        ax.minorticks_off()
        ax.set_title(TITLES[objective], fontsize=font_size + 1, pad=8)
        ax.set_xlabel('Embedding size', fontsize=font_size + 1)
        ax.set_ylabel('Regret AUC $\\downarrow$', fontsize=font_size + 1)
        ax.grid(True, ls='--', alpha=0.3)
    for ax in list(axes.flat)[len(objectives):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=True, fontsize=font_size)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')


def markdown(frame: pd.DataFrame) -> str:
    """A pipe table of ``frame`` (pandas' to_markdown needs the optional tabulate package)."""
    def fmt(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ''
        return f'{value:.4g}' if isinstance(value, float) else str(value)
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '|' + '---|' * len(frame.columns)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def main():
    df = pd.read_csv(os.path.join(ROOT, 'campaigns.tsv'), sep='\t')
    df['size'] = df['size'].astype(float)
    os.makedirs(OUT, exist_ok=True)
    summary = summarize(df)
    summary.to_csv(os.path.join(OUT, 'summary.tsv'), sep='\t', index=False)
    with open(os.path.join(OUT, 'summary.md'), 'w') as file:
        file.write(markdown(summary))
    result = tests(df)
    result.to_csv(os.path.join(OUT, 'tests.tsv'), sep='\t', index=False)
    with open(os.path.join(OUT, 'tests.md'), 'w') as file:
        file.write(markdown(result))
    figure(summary, os.path.join(OUT, 'auc_vs_size'))
    print(f'{len(df)} campaigns; wrote {OUT}/summary.*, tests.*, auc_vs_size.*')
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
