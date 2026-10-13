"""
Re-run of the HOPV15 and ClogP rows of ex_14, ex_19 and ex_25 after the target corrections of 2026-10-13.

Two errors in the targets were found during ex_25 (reviewer comment R3.3) and corrected (user decision: re-run HOPV15,
fix ClogP):

* HOPV15: the indices 5 ("PCE") and 4 ("Voc") selected two band gaps; the real PCE and Voc are the columns 3 and 0
  of the labels (see _slurm_ex_13.DATASETS). The analyses skip the archives with the old indices (target_matches).
* ClogP: mixin_clogp replaced the labels after the label scaler had been fitted on the logS labels of AqSolDB, so
  the models were trained on raw ClogP values and the reported errors were the true ones times the standard deviation
  of logS. predict_molecules.py now standardizes the labels after the after_dataset hook and marks its archives with
  "labels/scaled_after_hooks", which analyze_ex_25 requires for ClogP.

This file writes the affected runs under the prefixes of the original experiments, with their own command builders,
so that their analyses (analyze_ex_14, analyze_ex_19, analyze_ex_25) pick them up and skip the old runs:

* ex_14 (prefix ex_14_gnn): all variants on hopv15_pce (HDF, random and trained GIN / GATv2)
* ex_19 (prefix ex_19_fp): the five fingerprints on hopv15_pce
* ex_25 (prefix ex_25_desc): hopv15_pce (hdf_noglobal, morgan_desc; hdf and morgan come from ex_14 / ex_19 as
  before), hopv15_voc and clogp (all four arms)

All HDF runs use encodings that the original experiments have cached already (checked here, so the file has to be
written where the caches are), which is why the whole file can run as one queue of the retry stage:

    python _slurm_fix_targets.py      # writes _exfix/commands.txt, longest runs first
    CODE=$HOME/Programming/graph_hdc_ex14 EX14_STAGE=retry RETRY_FILE=_exfix/commands.txt sbatch --job-name=exfix --time=04:00:00 run_ex14_kcist.sbatch
"""
import os
import sys

import _slurm_ex_14 as ex14
import _slurm_ex_19 as ex19
import _slurm_ex_25 as ex25
from _slurm_ex_14 import params_of_line

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_exfix')
SEEDS = list(range(10))


def dataset(note: str) -> tuple:
    return next(ds for ds in ex25.DATASETS if ds[0] == note)


def cost(line: str) -> int:
    """Longest first: trained GNNs (1000 epochs), then ClogP (AqSolDB, ~10k molecules), then the HOPV15 runs."""
    if 'predict_molecules__gnn.py' in line:
        return 2
    return 1 if params_of_line(line)['NOTE'] == 'clogp' else 0


def build() -> list:
    lines = []
    hopv_pce = dataset('hopv15_pce')
    assert hopv_pce[2] == 3 and dataset('hopv15_voc')[2] == 0, 'HOPV15 indices are not the corrected ones'
    assert ex14.DATASETS[[ds[0] for ds in ex14.DATASETS].index('hopv15_pce')] == hopv_pce
    for seed in SEEDS:
        lines += list(ex14.variant_commands(hopv_pce, seed, ex14.PREFIX).values())
    lines += ex19.build(ex19.PREFIX, [hopv_pce], SEEDS)
    lines += [line for *_, line in ex25.runs(ex25.PREFIX, [hopv_pce, dataset('hopv15_voc'), dataset('clogp')], SEEDS)]
    # every HDF run must find its encoding cache, otherwise runs of the same key would race in the single queue
    missing = sorted({ex14.hdf_cache_name(params_of_line(line)) for line in lines
                      if 'predict_molecules__hdc.py' in line and not ex25.cache_files(line)})
    if missing:
        sys.exit('missing HDF encoding caches (run where the caches of ex_14 / ex_25 are):\n  ' + '\n  '.join(missing))
    return sorted(lines, key=lambda line: -cost(line))


if __name__ == '__main__':
    lines = build()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'commands.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    counts = {}
    for line in lines:
        params = params_of_line(line)
        key = (params['__PREFIX__'], params['NOTE'])
        counts[key] = counts.get(key, 0) + 1
    for (prefix, note), n in sorted(counts.items()):
        print(f'  {prefix:12s} {note:12s} {n:3d} runs')
    print(f'wrote {len(lines)} lines -> _exfix/commands.txt')
