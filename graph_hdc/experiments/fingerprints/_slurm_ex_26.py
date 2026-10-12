"""
Scheduler for Experiment 26 (ex_26): Crippen's logP as a further global attribute of HDF (GRAPH_CLOGP), to show that
the global attributes of HDF can be adapted to a dataset (response letter, opening reply: "Crippen's logP could easily
be added as a further global attribute for datasets that are known to benefit from it, as is the case for the
prediction of aqueous solubility").

The runs are the HDF arm of ex_14 round 4 (prefix ex_14_gnn: D = 2048, L = 2, continuous encoding, unit-modulus
codebooks, bidirectional message passing, total hydrogen counts, the ex_14 MLP and KNN) on the nine ex_14 targets and
seeds 0-9, with GRAPH_CLOGP = True as the only difference (the commands come from _slurm_ex_14.variant_commands). The
runs without logP are not repeated: analyze_ex_26.py pairs every run with the ex_14 round 4 run of the same target and
seed and checks that both used the same split.

The splits are fixed by the load__ caches of the KCIST main checkout, so the runs go through a pinned worktree whose
fingerprints/.cache holds links to the load__ and stats_ caches of the main checkout (not a link to the whole folder:
the new __clogp encoding caches stay in the worktree). run_ex14_kcist.sbatch (stage hdf) runs the command files:

    python _slurm_ex_26.py            # _ex26/hdf.txt + _ex26/hdf_primer_count.txt (90 runs, primer: one per dataset
                                      # and seed, which writes the encoding cache that the other QM9 targets reuse)
    python _slurm_ex_26.py smoke      # FreeSolv, seed 0: _ex26/smoke_hdf.txt + _ex26/smoke_hdf_primer_count.txt
    CODE=<worktree> EX14_STAGE=hdf HDF_FILE=_ex26/hdf.txt HDF_PRIMER_FILE=_ex26/hdf_primer_count.txt \
        sbatch --job-name=ex26 run_ex14_kcist.sbatch
"""
import os
import sys

from _slurm_ex_14 import DATASETS, variant_commands, params_of_line, is_current

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex26')

PREFIX = 'ex_26_clogp'
PREFIX_SMOKE = 'ex_26_smoke'
SEEDS = list(range(10))
# the only difference to the HDF commands of ex_14 round 4
EXTRA = {'GRAPH_CLOGP': True}


def hdf_line(ds: tuple, seed: int, prefix: str) -> str:
    """The ex_14 round 4 HDF command of (dataset, seed) with logP as a further global attribute."""
    line = variant_commands(ds, seed, prefix, overrides={'hdf': EXTRA})['hdf']
    params = params_of_line(line)
    assert is_current('hdc', params) and params['GRAPH_CLOGP'] is True, line
    return line


def build(datasets: list, seeds: list, prefix: str) -> tuple:
    """(primer, rest): the primer block has one run per (DATASET_NAME, seed); the rest reuse its encoding cache."""
    primer, rest, primed = [], [], set()
    for seed in seeds:
        for ds in datasets:
            key = (ds[1], seed)
            (rest if key in primed else primer).append(hdf_line(ds, seed, prefix))
            primed.add(key)
    return primer, rest


def write(name: str, lines: list):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'wrote {len(lines):4d} lines -> _ex26/{name}')


if __name__ == '__main__':
    if 'smoke' in sys.argv[1:]:
        primer, rest = build([d for d in DATASETS if d[0] == 'freesolv_hfe'], [0], PREFIX_SMOKE)
        stem = 'smoke_'
    else:
        primer, rest = build(DATASETS, SEEDS, PREFIX)
        stem = ''
    write(f'{stem}hdf.txt', primer + rest)
    write(f'{stem}hdf_primer_count.txt', [str(len(primer))])
