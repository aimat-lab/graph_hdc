"""
Scheduler for the revision re-run of Experiment 08 (ex_08_b): edit-based GED correlation (reviewer comment R2.7).

Re-runs the edit-based analysis of the main text (Figure 3a; original runs: prefix ex_08_a, ``_slurm_ex_08.py``)
with the same configuration: 100 seed molecules from ZINC250k, 20 molecules at each of 1-3 edit steps, Morgan
fingerprints (radius 2, binary) and HDF (depth 2, continuous) at 32, 128, 512 and 2048 dimensions. New compared
to ex_08_a:

- every run stores the distance of every generated molecule to its seed molecule (``ged_pairs.csv``), for the
  distances-per-edit-step figure of the SI;
- the seed SMILES is canonicalized before its neighborhood is generated (``GED_CANONICAL_QUERY``), so that HDF
  and Morgan are evaluated on identical molecule pairs. In ex_08_a, the Morgan runs used the raw ZINC250k
  SMILES, which end with a newline, whereas the HDF runs used canonical SMILES. As a consequence, the seed was
  never recognized as already visited in the Morgan runs, so that in about 15 of the 100 seeds it reappeared as
  its own neighbor at two or three edit steps (Tanimoto distance 0; 16 seeds with similarity 1.0 at D >= 128),
  and 3 seeds had differently written SMILES and hence different neighborhoods than in the HDF runs. The Morgan
  numbers of ex_08_b therefore differ from ex_08_a for reasons unrelated to HDF;
- HDF uses the revision settings: bidirectional message passing and the total hydrogen count.

Two prefixes:

- ``ex_08_b``: Morgan and HDF with the revision settings (the new results; Figure 3a is regenerated from them).
- ``ex_08_b_repro``: reproduction check against ex_08_a, i.e. HDF with the old settings (one-directional,
  implicit hydrogens) and Morgan without canonicalized seeds. These should reproduce the per-seed correlations
  of ex_08_a (``ged_correlation_summary.csv``) exactly.

Each job runs one experiment on euler with its own TMPDIR and NUM_THREADS threads (config euler_3).

    python _slurm_ex_08b.py             # pre-load the dataset, write job scripts, submit
    python _slurm_ex_08b.py --dry-run   # write job scripts only
    python _slurm_ex_08b.py --smoke     # write small smoke-test job scripts (prefix ex_08_b_smoke), print paths
"""
import os
import sys

PATH = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.abspath(os.path.join(PATH, '..', '..', '..', '.venv'))
OUT = os.path.join(PATH, '_ex08b')

AUTOSLURM_CONFIG = 'euler_3'
CPUS = 4
NUM_THREADS = 2
EMBEDDING_SIZES = [32, 128, 512, 2048]

# same configuration as ex_08_a (see _slurm_ex_08.py)
FIXED = {
    '__DEBUG__': False,
    '__CACHING__': False,
    'DATASET_NAME': 'zinc250k',
    'SEED': 1,
    'NUM_SAMPLES': 10,
    'NUM_NEIGHBORS': 5,
    'FIND_DISSIMILAR': True,
    'ENABLE_GED_ANALYSIS': True,
    'GED_NUM_SAMPLES': 100,
    'NUM_HOPS': 3,
    'NUM_NEIGHBOR_BRANCHES': 5,
    'NUM_NEIGHBOR_TOTAL': 20,
}

FP = {'FINGERPRINT_TYPE': 'morgan', 'FINGERPRINT_RADIUS': 2, 'USE_COUNTS': False}
HDC = {'NUM_LAYERS': 2, 'ENCODING_MODE': 'continuous', 'DEVICE': 'cpu'}
SIZE_PARAMETER = {'fp': 'FINGERPRINT_SIZE', 'hdc': 'EMBEDDING_SIZE'}

# (prefix, encoding, extra parameters, job time limit). HDF encodes all ~250k ZINC250k molecules: about 32-37 ms
# per molecule at D=2048 on 2 threads (review measurement), so about 3 h in total; 12 h leaves room on the
# loaded machine.
ARMS = [
    # both HDF arms ran with the original (gaussian) codebooks; unit-modulus codebooks became the default later
    ('ex_08_b', 'hdc', {**HDC, 'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'GED_CANONICAL_QUERY': True,
                        'SPECTRUM': 'gaussian'}, '12:00:00'),
    ('ex_08_b', 'fp', {**FP, 'GED_CANONICAL_QUERY': True}, '02:00:00'),
    ('ex_08_b_repro', 'hdc', {**HDC, 'BIDIRECTIONAL': False, 'HYDROGEN_COUNT': 'implicit', 'GED_CANONICAL_QUERY': True,
                              'SPECTRUM': 'gaussian'}, '12:00:00'),
    ('ex_08_b_repro', 'fp', {**FP, 'GED_CANONICAL_QUERY': False}, '02:00:00'),
]


def memory(encoding: str, size: int) -> str:
    """Memory per job: HDF keeps the embeddings of all molecules (about 28 GB peak at D=2048, review measurement)."""
    if encoding == 'hdc':
        return '40000mb' if size >= 2048 else '24000mb'
    return '16000mb'

# small versions for the local smoke test: subsampled dataset, few queries
SMOKE_OVERRIDES = {'NUM_DATA': 2000, 'NUM_SAMPLES': 2, 'GED_NUM_SAMPLES': 2}


def command(prefix: str, encoding: str, extra: dict, size: int, overrides: dict = None) -> str:
    """One ``python molecule_similarity__<encoding>.py --K="repr(v)" ...`` line (same format as _slurm_ex_08.py)."""
    params = {'__PREFIX__': prefix, **FIXED, **extra, SIZE_PARAMETER[encoding]: size,
              'DATASET_NAME_ID': f"{FIXED['DATASET_NAME']}_sim_ged_{encoding}_s{size}"}
    params.update(overrides or {})
    return ' '.join(['python', f'molecule_similarity__{encoding}.py'] + [f'--{k}="{v!r}"' for k, v in params.items()])


def write_job(name: str, line: str) -> str:
    """One bash script per SLURM job: CPU only, NUM_THREADS threads, private TMPDIR."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('set -e\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write(f'export OMP_NUM_THREADS={NUM_THREADS} MKL_NUM_THREADS={NUM_THREADS} OPENBLAS_NUM_THREADS={NUM_THREADS}\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def jobs(smoke: bool = False) -> list:
    """(job script path, time limit, memory) for every arm and size."""
    result = []
    for prefix, encoding, extra, job_time in ARMS:
        if smoke:
            prefix = prefix.replace('ex_08_b', 'ex_08_b_smoke')
        for size in ([32] if smoke else EMBEDDING_SIZES):
            name = f'{prefix}__{encoding}__{size}'
            line = command(prefix, encoding, extra, size, SMOKE_OVERRIDES if smoke else None)
            result.append((write_job(name, line), job_time, memory(encoding, size)))
    return result


def preload():
    """Download the graph dataset to /tmp once before submitting (same call as molecule_similarity.py)."""
    from chem_mat_data.main import load_graph_dataset
    graphs = load_graph_dataset(FIXED['DATASET_NAME'], folder_path='/tmp')
    print(f"pre-loaded {FIXED['DATASET_NAME']}: {len(graphs)} graphs")


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        for path, _, _ in jobs(smoke=True):
            print(path)
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter
    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload()
    all_jobs = jobs()
    # one submitter per (time limit, memory), since both are fillers of the job template
    for job_time, mem in sorted({(t, m) for _, t, m in all_jobs}, reverse=True):
        submitter = ASlurmSubmitter(
            config_name=AUTOSLURM_CONFIG,
            batch_size=1,
            randomize=False,
            dry_run=dry_run,
            overwrite_fillers={
                'venv': VENV,
                'cwd': PATH,
                'time': job_time,
                'cpus': str(CPUS),
                'num_threads': str(NUM_THREADS),
                'mem': mem,
                'job_name': 'ex08b_ged',
                'additional_sbatch_configs': '#SBATCH --nodelist=euler',
            },
            archive_path=PATH,
        )
        for path, t, m in all_jobs:
            if (t, m) == (job_time, mem):
                submitter.add_command(f'bash {path}')
        print(f'time {job_time}, mem {mem}: {submitter.count_jobs()} job(s), config {AUTOSLURM_CONFIG}')
        submitter.submit()
