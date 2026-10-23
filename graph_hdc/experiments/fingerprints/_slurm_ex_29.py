"""
Command-file generator for Experiment 29 (ex_29): the three figures of the paper that still came from the original HDF
encoder, re-run with the standard encoder on JUPITER (user decisions 2026-10-23). Each keeps the protocol of its
original sweep; only the HDF encoder changes and every figure gets 10 repetitions.

* Figure 4 (fig:ex_size; prefix ex_29_size; originally ex_03_ae and ex_03b_aa): the test MAE of the MLP of Table 1
  (neural_net2, (100, 100), learning rate 1e-3, 200 epochs, batch size 64) over the dimension (8 - 16,384) for HDF
  (depth 1 and 3) and Morgan fingerprints (radius 1 and 3) on AqSolDB (logS) and BACE (pIC50).
* Figure 3b (fig:ex_bio b; prefix ex_29_knn; originally ex_10_b): the KNN error ratio MAE_Morgan / MAE_HDF over the
  dimension (32 - 2048) at depth / radius 2 with k = 5 and uniform weights, on FreeSolv, Lipophilicity, AqSolDB and the
  QM9 gap. Computed twice: with the native distance of each representation (Jaccard for Morgan, cosine for HDF, as in
  Table S6 and Figure 2) and with the Euclidean distance for both (as the published panel); the user picks one. (For
  the unit-length HDF vectors the Euclidean distance orders the neighbors like the cosine distance up to ties between
  identical vectors; the review measured MAE differences below 0.1%. The HDF runs are made with both distances anyway,
  so that the Euclidean variant uses the Euclidean distance for both representations.)
* SI Figure S1 (fig:ex04; prefix ex_29_train; originally ex_04): the test MAE of the scikit-learn MLP (neural_net:
  (100, 100), early stopping) over the number of training molecules (10 - 100,000) for HDF and Morgan fingerprints at
  depth / radius 1-3 and 2048 dimensions, on the QM9 targets U0, HOMO-LUMO gap, heat capacity Cv and dipole moment
  (one run per configuration, TARGET_INDICES). Table 1's MLP cannot train on fewer than 64 molecules (it drops
  incomplete batches of 64), so S1 keeps the network of its original sweep.

Common settings: the seeds 0-9 (split, network and, for HDF, the codebooks: CODEBOOK_SEED None draws new codebooks in
every repetition, as the original sweeps did), the standard HDF encoder (bidirectional message passing, unit-modulus
codebooks, total hydrogen counts, graph size and diameter), the 80/10/10 splits of predict_molecules with the dataset
caches (molecule order) of the ex_27 worktree. The HDF encoding runs in chunks of max(16, 600 * 2048 / D) molecules
(HDC_ENCODE_BATCH; the encodings do not depend on it), which keeps the peak memory of a run at a few GB also at 16,384
dimensions (review 2026-10-23: 18 GB for one chunk of 600 AqSolDB molecules at 16,384 dimensions and depth 3).

Caching: the Figure 4 and 3b encodings serve one run each (new codebooks per seed) and are not cached (HDC_CACHE
False). The 30 S1 encodings of QM9 (10 seeds x 3 depths, 2048 dimensions) serve 13 training sizes each: the stage
writes them first (warmup barrier, about 2 GB each) and the S1 runs read them (HDC_CACHE True).

Two stages of run_ex27.sbatch, each a whole node, submitted side by side (submit_ex29.sh). JUPITER Booster production
jobs must use all 4 GPUs, so both stages have GPU work:
* ex29g: Figure 4 on the GPU pool (GPU_J slots over the 4 GPUs; the HDF encodings run on the CPU threads of a slot, so
  the slots get many threads: GPU_J=24, GPU_THREADS=12). File ex29g_gpu.txt.
* ex29c: SI Figure S1 on the CPU pool: the QM9 encodings as the warmup barrier (ex29c_warmup.txt), then the 780 runs
  (ex29c_cpu.txt), HDF and Morgan alternating and the largest training sets first (CPU_J=30, CPU_THREADS=6); and
  Figure 3b on the GPU pool (ex29c_gpu.txt; GPU_J=12, GPU_THREADS=9).
Memory: the node has 480 GB of CPU memory, but "available" also counts the 4 x 96 GB of GPU memory, so MEMFREE=400G.
A cache counts as written only if a completed warmup archive of its seed and depth exists and the cache file is
complete (pycomex writes key.pkl, compresses it to key.pkl.gz and then deletes key.pkl); cache files without that
(e.g. from a warmup killed while writing) are listed and the ex29c files are not written until they are removed.

Usage (from the fingerprints folder of the ex_27 worktree on JUPITER):
    python _slurm_ex_29.py [g|c] # writes the files of stage ex29g, ex29c or both: only the runs without a completed
                                 # archive and the warmups of encodings that are not cached yet
    sbatch --export=ALL,EX27_STAGE=ex29g,GPU_J=24,GPU_THREADS=12,MEMFREE=400G run_ex27.sbatch
    sbatch --export=ALL,EX27_STAGE=ex29c,CPU_J=30,CPU_THREADS=6,GPU_J=12,GPU_THREADS=9,MEMFREE=400G run_ex27.sbatch
    python analyze_ex_29.py      # the CSVs of the three figures in _ex27/ex29/
"""
import os
import sys
import glob

from _slurm_ex_14 import params_of_line
from _slurm_ex_27 import PATH, HDF, METHOD, MLP, KNN, SEEDS, _command, group_of_note
from _slurm_ex_28 import completed, line_is_done, interleave, _write

PREFIX_SIZE = 'ex_29_size'
PREFIX_KNN = 'ex_29_knn'
PREFIX_TRAIN = 'ex_29_train'
PREFIX_WARMUP = 'ex_29_warmup'

# the standard encoder with new codebooks in every repetition, each encoding computed in its run
HDF_29 = {**HDF, 'CODEBOOK_SEED': None, 'HDC_CACHE': False}

# Figure 4
SIZE_NOTES = ['aqsoldb_logs', 'bace_ic50']
SIZE_DIMENSIONS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
SIZE_DEPTHS = [1, 3]

# Figure 3b: {variant: {method: KNN metric}}; 'minkowski' (p = 2) is the Euclidean distance
KNN_NOTES = ['freesolv_hfe', 'lipophilicity_logD', 'aqsoldb_logs', 'qm9_gap']
KNN_DIMENSIONS = [32, 64, 128, 256, 512, 1024, 2048]
KNN_DEPTH = 2
KNN_VARIANTS = {'native': {'hdf': METHOD['hdf'][2], 'morgan': METHOD['morgan'][2]},
                'euclidean': {'hdf': 'minkowski', 'morgan': 'minkowski'}}

# SI Figure S1
TRAIN_NOTES = ['qm9_energy', 'qm9_gap', 'qm9_cv', 'qm9_dipole']
TRAIN_SIZES = [10, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10_000, 25_000, 50_000, 100_000]
TRAIN_DEPTHS = [1, 2, 3]
TRAIN_DIMENSION = 2048
SKLEARN_MLP = {'NN_HIDDEN_LAYER_SIZES': (100, 100)}

METHODS_29 = ['hdf', 'morgan']


def encode_batch(dimension: int) -> int:
    """Molecules per chunk of the HDF encoding: 600 at 2048 dimensions, fewer above and more below (at most 4800:
    disentangling the results of a chunk costs time proportional to its size per molecule)."""
    return min(4800, max(16, 600 * 2048 // dimension))


def representation(method: str, dimension: int, depth: int) -> dict:
    """The featurization parameters of HDF (dimension, message-passing layers) or Morgan (bits, radius)."""
    if method == 'hdf':
        return {**HDF_29, 'EMBEDDING_SIZE': dimension, 'NUM_LAYERS': depth,
                'HDC_ENCODE_BATCH': encode_batch(dimension)}
    return {'FINGERPRINT_TYPE': 'morgan', 'FINGERPRINT_SIZE': dimension, 'FINGERPRINT_RADIUS': depth}


def size_command(prefix: str, seed: int, note: str, method: str, dimension: int, depth: int) -> str:
    """Figure 4: the MLP of Table 1 on one target (data on the device of the slot's GPU)."""
    return _command(METHOD[method][0], prefix, seed, group_of_note(note), [note],
                    {**representation(method, dimension, depth), 'MODELS': ['neural_net2'], **MLP,
                     'NN_DEVICE_DATA': True})


def knn_command(prefix: str, seed: int, note: str, method: str, dimension: int, metric: str) -> str:
    """Figure 3b: the KNN (k = 5, uniform weights, torch backend) with the given metric on one target."""
    return _command(METHOD[method][0], prefix, seed, group_of_note(note), [note],
                    {**representation(method, dimension, KNN_DEPTH), 'MODELS': ['k_neighbors'], **KNN,
                     'KN_METRIC': metric})


def train_command(prefix: str, seed: int, notes: list, method: str, depth: int, num_train: int,
                  cache: bool = True) -> str:
    """SI Figure S1: the scikit-learn MLP on NUM_TRAIN training molecules, the targets of one group in one run. The
    HDF runs read the encoding written by the warmup of their seed and depth (``cache``)."""
    params = representation(method, TRAIN_DIMENSION, depth)
    if method == 'hdf':
        params['HDC_CACHE'] = cache
    return _command(METHOD[method][0], prefix, seed, group_of_note(notes[0]), notes,
                    {**params, 'MODELS': ['neural_net'], **SKLEARN_MLP, 'NUM_TRAIN': num_train}, multi=True)


def train_warmup_command(seed: int, depth: int) -> str:
    """A run without models that writes the HDF encoding cache of QM9 for a seed (codebook) and depth."""
    return _command('hdc', PREFIX_WARMUP, seed, group_of_note(TRAIN_NOTES[0]), TRAIN_NOTES[:1],
                    {**representation('hdf', TRAIN_DIMENSION, depth), 'HDC_CACHE': True, 'MODELS': []})


def train_cache_files(seed: int, depth: int) -> list:
    """The files of the QM9 encoding cache of a seed and depth (key of predict_molecules__hdc for CODEBOOK_SEED None,
    NUM_DATA 1.0): key.pkl.gz when complete, key.pkl while it is written."""
    key = f'hdc_qm9_smiles__seed_{seed}__size_{TRAIN_DIMENSION}__depth_{depth}__bidir__totalh__unitspec'
    return sorted(glob.glob(os.path.join(PATH, '.cache', key + '.pkl*')))


def train_cache_state(warmed: set) -> tuple:
    """(complete, stale): the (seed, depth) whose cache is complete (a completed warmup archive and only key.pkl.gz),
    and the cache files of the others (left by an interrupted warmup, or of unknown origin)."""
    complete, stale = set(), []
    for seed in SEEDS:
        for depth in TRAIN_DEPTHS:
            files = train_cache_files(seed, depth)
            if (seed, depth) in warmed and len(files) == 1 and files[0].endswith('.pkl.gz'):
                complete.add((seed, depth))
            else:
                stale += files
    return complete, stale


def warmed_up() -> set:
    """The (seed, depth) of the completed warmup archives (the encodings they wrote)."""
    return {(p['SEED'], p['NUM_LAYERS']) for module, p in completed(PREFIX_WARMUP) if module == 'hdc'
            and p.get('DATASET_NAME') == 'qm9_smiles' and p.get('EMBEDDING_SIZE') == TRAIN_DIMENSION
            and p.get('CODEBOOK_SEED') is None and p.get('NUM_DATA') == 1.0}


def size_lines() -> list:
    return [size_command(PREFIX_SIZE, seed, note, method, dimension, depth)
            for note in SIZE_NOTES for method in METHODS_29 for dimension in SIZE_DIMENSIONS
            for depth in SIZE_DEPTHS for seed in SEEDS]


def knn_lines() -> list:
    return [knn_command(PREFIX_KNN, seed, note, method, dimension, metric)
            for metrics in KNN_VARIANTS.values() for note in KNN_NOTES for method, metric in metrics.items()
            for dimension in KNN_DIMENSIONS for seed in SEEDS]


def train_lines() -> list:
    return [train_command(PREFIX_TRAIN, seed, TRAIN_NOTES, method, depth, num_train)
            for method in METHODS_29 for depth in TRAIN_DEPTHS for num_train in TRAIN_SIZES for seed in SEEDS]


def gpu_cost(line: str) -> int:
    """Rough cost class of a GPU line (higher = longer, more memory): HDF encodings of QM9 (134k molecules) and the
    HDF runs at 4096 dimensions and more (AqSolDB and BACE, whose molecules are larger) first."""
    hdf = 'predict_molecules__hdc.py' in line
    if 'qm9_smiles' in line:
        return 3 if hdf else 2
    if hdf and any(f'--EMBEDDING_SIZE="{d}"' in line for d in (4096, 8192, 16384)):
        return 2
    return 0


def train_cost(line: str) -> int:
    """The S1 runs with the most training molecules first (the scikit-learn MLP takes longest there)."""
    return next(n for n in TRAIN_SIZES if f'--NUM_TRAIN="{n}"' in line)


def spread(lines: list, key) -> list:
    """The lines ordered by -key, with the lines of equal key alternating between HDF and Morgan."""
    out = []
    for value in sorted({key(line) for line in lines}, reverse=True):
        group = [line for line in lines if key(line) == value]
        out += interleave([l for l in group if 'predict_molecules__hdc.py' in l],
                          [l for l in group if 'predict_molecules__hdc.py' not in l])
    return out


def gpu_order(lines: list) -> list:
    """The heavy GPU lines (by cost class, HDF and Morgan alternating) interleaved with the light ones, so that the
    heavy runs start early but not all at the same time."""
    heavy = spread([line for line in lines if gpu_cost(line) > 0], gpu_cost)
    light = [line for line in lines if gpu_cost(line) == 0]
    return interleave(heavy, light)


if __name__ == '__main__':
    # the stage whose files are written (g, c or both): never rewrite the file of a stage whose job is running
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage not in ('g', 'c', 'all'):
        sys.exit(f'unknown stage {stage} (g, c or all)')
    sizes, knns, trains = size_lines(), knn_lines(), train_lines()
    assert len(sizes) == len(SIZE_NOTES) * 2 * len(SIZE_DIMENSIONS) * len(SIZE_DEPTHS) * len(SEEDS)
    assert len(knns) == sum(map(len, KNN_VARIANTS.values())) * len(KNN_NOTES) * len(KNN_DIMENSIONS) * len(SEEDS)
    assert len(trains) == 2 * len(TRAIN_DEPTHS) * len(TRAIN_SIZES) * len(SEEDS)
    assert len(set(sizes + knns + trains)) == len(sizes) + len(knns) + len(trains)
    done = {prefix: completed(prefix) for prefix in (PREFIX_SIZE, PREFIX_KNN, PREFIX_TRAIN)}
    gpu_g = [line for line in sizes if not line_is_done(line, done[PREFIX_SIZE])]
    gpu_c = [line for line in knns if not line_is_done(line, done[PREFIX_KNN])]
    cpu = [line for line in trains if not line_is_done(line, done[PREFIX_TRAIN])]
    print(f'Figure 4: {len(sizes)} runs, Figure 3b: {len(knns)} runs, S1: {len(trains)} runs; without a completed '
          f'archive: {len(gpu_g)} (ex29g GPU), {len(gpu_c)} (ex29c GPU), {len(cpu)} (ex29c CPU)')
    if stage in ('g', 'all'):
        _write('ex29g_gpu.txt', gpu_order(gpu_g))
    if stage in ('c', 'all'):
        complete, stale = train_cache_state(warmed_up())
        if stale:
            sys.exit('!!! QM9 encoding cache files without a completed warmup (an interrupted warmup?); remove them '
                     'before writing the ex29c files:\n' + '\n'.join(stale))
        # warmups only for the (seed, depth) of pending HDF runs whose encoding is not cached yet
        pending = {(p['SEED'], p['NUM_LAYERS']) for p in map(params_of_line, cpu)
                   if 'EMBEDDING_SIZE' in p}
        warmup = [train_warmup_command(seed, depth) for seed in SEEDS for depth in TRAIN_DEPTHS
                  if (seed, depth) in pending - complete]
        _write('ex29c_warmup.txt', warmup)
        _write('ex29c_cpu.txt', spread(cpu, train_cost))
        _write('ex29c_gpu.txt', gpu_order(gpu_c))
        print(f'{len(complete)} S1 encodings cached, {len(warmup)} to write')
