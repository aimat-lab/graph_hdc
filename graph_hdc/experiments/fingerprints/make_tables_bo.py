"""
LaTeX tables of the Bayesian optimization benchmark ex_24 for the Supplementary Information.

- ``bo_hdf_linear.tex``: median regret AUC of HDF with the linear (cosine) kernel next to the Matern kernel of
  the main comparison, per objective and embedding size (all objectives except QED on ZINC250k).
- ``bo_runtime.tex``: median wall-clock seconds of a complete campaign on ZINC250k (QED objective), per
  representation and embedding size.

Reads ``_ex24/campaigns.tsv`` (``bo_benchmark.py collect``; EX24_ROOT points elsewhere).

    python make_tables_bo.py <output folder>
"""
import os
import sys

import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EX24_ROOT', os.path.join(PATH, '_ex24'))
SIZES = [8, 16, 32, 64, 128, 512, 1024]
OBJECTIVES = [
    ('clogp', 'ClogP'), ('solubility', 'Solubility'), ('lipophilicity', 'Lipophilicity'),
    ('freesolv', 'Hydration free energy'), ('beet', 'Bee toxicity'), ('qm9_gap', 'HOMO--LUMO gap'),
    ('qm9_dipole', 'Dipole moment'), ('melting_point', 'Melting point'),
]
RUNTIME_METHODS = [('morgan_tanimoto', 'Morgan FP (Tanimoto)'), ('morgan_pca', 'Morgan FP, PCA (Mat\\\'{e}rn)'),
                   ('hdf_matern', 'HDF (Mat\\\'{e}rn)')]


def hdf_linear_table(df: pd.DataFrame) -> str:
    medians = df[df.method.isin(['hdf_linear', 'hdf_matern'])].groupby(['objective', 'method', 'size']).auc.median()
    lines = [
        '\\begin{tabular}{llrrrrrrr}',
        '\\toprule',
        '\\textbf{Objective} & \\textbf{Kernel} & ' + ' & '.join(f'\\textbf{{{s}}}' for s in SIZES) + ' \\\\',
        '\\midrule',
    ]
    for i, (objective, name) in enumerate(OBJECTIVES):
        rows = {}
        for method in ('hdf_linear', 'hdf_matern'):
            rows[method] = [medians.get((objective, method, float(s)), float('nan')) for s in SIZES]
        for method, label in (('hdf_linear', 'linear'), ('hdf_matern', 'Mat\\\'{e}rn')):
            cells = []
            for j, value in enumerate(rows[method]):
                other = rows['hdf_matern' if method == 'hdf_linear' else 'hdf_linear'][j]
                # compare the printed values: equal-looking numbers are a tie, and a tie is not marked
                text, other_text = f'{value:.3f}', f'{other:.3f}'
                cells.append(f'\\textbf{{{text}}}' if float(text) < float(other_text) else text)
            first = name if method == 'hdf_linear' else ''
            lines.append(f'{first} & {label} & ' + ' & '.join(cells) + ' \\\\')
        if i < len(OBJECTIVES) - 1:
            lines.append('\\addlinespace')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def runtime_table(df: pd.DataFrame) -> str:
    seconds = df[df.objective == 'qed'].groupby(['method', 'size']).seconds_total.median()
    lines = [
        '\\begin{tabular}{rrrr}',
        '\\toprule',
        '$\\boldsymbol{D}$ & ' + ' & '.join(f'\\textbf{{{label}}}' for _, label in RUNTIME_METHODS) + ' \\\\',
        '\\midrule',
    ]
    for size in SIZES:
        cells = [f'{seconds[(method, float(size))]:.0f}' for method, _ in RUNTIME_METHODS]
        lines.append(f'{size} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'analysis')
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, 'campaigns.tsv'), sep='\t')
    df['size'] = df['size'].astype(float)
    with open(os.path.join(out, 'bo_hdf_linear.tex'), 'w') as file:
        file.write(hdf_linear_table(df))
    with open(os.path.join(out, 'bo_runtime.tex'), 'w') as file:
        file.write(runtime_table(df))
    print(f'wrote {out}/bo_hdf_linear.tex and bo_runtime.tex')


if __name__ == '__main__':
    main()
