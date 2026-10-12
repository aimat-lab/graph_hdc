"""
Scheduler for Experiment 22 (ex_22): unit-modulus spectra and the graph size/diameter encodings of HDF.

The random codebook vectors of HDF (element vectors and the base vectors of the fractional power encoders) have
random Fourier magnitudes. Binding by circular convolution multiplies these magnitudes, so a few Fourier
components dominate the embeddings (about 10-100 effective components at any D for the structure part). The
``SPECTRUM="unit"`` option of predict_molecules__hdc.py gives all codebook vectors unit Fourier magnitudes with
unchanged phases (see graph_hdc/special/spectrum.py). In the exact-GED analysis of ex_17 (R2.7), unit-modulus spectra and
removing the graph size and diameter encodings both raised the correlation with the GED; this experiment measures
how the two changes affect property prediction.

Four arms, 2 x 2 (user decision 2026-10-09):

    SPECTRUM      GRAPH_ATTRIBUTES
    gaussian      True               the original HDF (bidirectional, total hydrogen counts)
    unit          True
    gaussian      False              without size/diameter: the readout normalized to unit length
    unit          False

HDF D=2048, L=2; MLP only (neural_net2, (100, 100), lr 1e-3, as in Table 1 and ex_15/ex_20); the five small
regression targets of ex_15; seeds 0-9. All arms of a seed use the same split, the same random draws for the
codebooks (the unit-modulus vectors keep their phases) and the same MLP seed, so ``analyze_ex_22.py`` can compare
the arms seed by seed. Each SLURM job runs one group script from ``_ex22/jobs/`` (the four arms of one dataset and
seed, one after the other). Job setup as in ex_15/ex_20: CPU only, 2 threads, 4 CPUs per job, private TMPDIR,
caching disabled (no job writes shared cache files), datasets pre-loaded to /tmp before submitting. The group
scripts put the repository that contains this file first on PYTHONPATH, so a run from a git worktree uses the
worktree's graph_hdc package (the venv's editable install points to the main checkout).

    python _slurm_ex_22.py             # pre-load datasets, write group scripts, submit
    python _slurm_ex_22.py --dry-run   # write group scripts and job scripts only
    python _slurm_ex_22.py --smoke     # write one group script (FreeSolv, seed 0, prefix ex_22_smoke) and print
                                       # its path, to run it locally with bash
"""
import os
import sys

from _slurm_ex_14 import DATASETS, VARIANTS, _command
from _slurm_ex_15 import preload

PATH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(PATH, '..', '..', '..'))
OUT = os.path.join(PATH, '_ex22')
# the venv lives in the main checkout only (a worktree has none)
VENV = os.path.join(REPO, '.venv')
if not os.path.isdir(VENV):
    VENV = '/media/ssd2/Programming/graph_hdc/.venv'

PREFIX = 'ex_22_unitspec'
PREFIX_SMOKE = 'ex_22_smoke'
SEEDS = list(range(10))
# (SPECTRUM, GRAPH_ATTRIBUTES); the first arm is the original HDF
ARMS = [('gaussian', True), ('unit', True), ('gaussian', False), ('unit', False)]
# the five small experimental targets of ex_14/ex_15
SELECTED = ['freesolv_hfe', 'aqsoldb_logs', 'lipophilicity_logD', 'bace_ic50', 'hopv15_pce']
DATASETS_22 = [ds for ds in DATASETS if ds[0] in SELECTED]

AUTOSLURM_CONFIG = 'euler_3'
NUM_THREADS = 2
CPUS = 4


def commands(prefix: str, ds: tuple, seed: int) -> list:
    module, params = VARIANTS['hdf']
    params = {**params, 'MODELS': ['neural_net2'], '__CACHING__': False,
              'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'SPECTRUM_DIAGNOSTIC': True}
    return [_command(module, prefix, seed, ds, {**params, 'SPECTRUM': spectrum, 'GRAPH_ATTRIBUTES': attributes})
            for spectrum, attributes in ARMS]


def write_group(name: str, lines: list) -> str:
    """One bash script per SLURM job: CPU only, private TMPDIR, this repository first on PYTHONPATH."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write(f'export OMP_NUM_THREADS={NUM_THREADS} MKL_NUM_THREADS={NUM_THREADS} OPENBLAS_NUM_THREADS={NUM_THREADS}\n')
        f.write(f'export PYTHONPATH="{REPO}${{PYTHONPATH:+:$PYTHONPATH}}"\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        f.write('python -c "import graph_hdc; print(\'graph_hdc from\', graph_hdc.__file__)"\n')
        f.write(f'git -C "{REPO}" rev-parse HEAD\n')
        for line in lines:
            f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def groups(prefix: str, datasets: list, seeds: list) -> list:
    return [write_group(f'{prefix}__{ds[0]}__seed_{seed}', commands(prefix, ds, seed))
            for ds in datasets for seed in seeds]


if __name__ == '__main__':

    if '--smoke' in sys.argv:
        ds = [d for d in DATASETS_22 if d[0] == 'freesolv_hfe']
        print(groups(PREFIX_SMOKE, ds, [0])[0])
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter

    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload(DATASETS_22)

    paths = groups(PREFIX, DATASETS_22, SEEDS)
    submitter = ASlurmSubmitter(
        config_name=AUTOSLURM_CONFIG,
        batch_size=1,
        randomize=False,
        dry_run=dry_run,
        overwrite_fillers={
            'venv': VENV,
            'cwd': PATH,
            'time': '12:00:00',
            'cpus': str(CPUS),
            'num_threads': str(NUM_THREADS),
            'mem': '8000mb',
            'job_name': 'ex22_unitspec',
            'additional_sbatch_configs': '#SBATCH --nodelist=euler',
        },
        archive_path=PATH,
    )
    for path in paths:
        submitter.add_command(f'bash {path}')

    runs = len(ARMS) * len(SEEDS) * len(DATASETS_22)
    print(f'{runs} runs ({len(DATASETS_22)} datasets x {len(SEEDS)} seeds x {len(ARMS)} arms) in '
          f'{submitter.count_jobs()} job(s), config {AUTOSLURM_CONFIG}, venv {VENV}')
    submitter.submit()
