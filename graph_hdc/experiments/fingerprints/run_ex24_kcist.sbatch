#!/bin/bash
# Whole-node runner for Experiment 24 (BO benchmark, bo_benchmark.py) on the KCIST cluster ("small" partition
# nodes: 64 CPU, 4x RTX 4090, 478 GB). One stage per job (EX24_STAGE):
#
#   featurize  clean the libraries, Morgan bits, PCA, objectives, initial designs (CPU only)
#   encode     HDF embeddings of every library, size and campaign codebook (CPU only)
#   calibrate  per-round timings and node throughput for several worker counts (CPU + GPU)
#   run        BO campaigns of one shard of _ex24/tasks.jsonl (CPU + GPU acquisition for ZINC250k). Submit as
#              a job array, one shard per node; the array MUST be 0-based and contiguous (--array=0-3).
#
# Variables come from the submitting environment. Do NOT use --export=ALL,VAR=...: on KCIST that makes most nodes
# fail the job start with "user env retrieval failed requeued held" (sbatch exports the environment by default).
# Submit from the fingerprints folder of the pinned checkout (CODE). The log folder _ex24/ must exist before the
# first submission (it is git-ignored, and SLURM does not create the folder of --output):
#
#   mkdir -p _ex24
#   EX24_STAGE=featurize WORKERS=64 sbatch --job-name=ex24feat --gres=gpu:0 --time=02:00:00 run_ex24_kcist.sbatch
#   EX24_STAGE=encode WORKERS=32 THREADS=2 sbatch --job-name=ex24enc --gres=gpu:0 --dependency=afterok:<feat> run_ex24_kcist.sbatch
#     (each encode worker holds the PyG graphs of the library, ~5 GB for ZINC250k: keep WORKERS at about 32)
#   EX24_STAGE=calibrate sbatch --job-name=ex24cal --dependency=afterok:<enc> --time=03:00:00 run_ex24_kcist.sbatch
#   python bo_benchmark.py tasks          # on the login node, after featurize
#   EX24_STAGE=run WORKERS=64 THREADS=1 sbatch --job-name=ex24run --array=0-3 run_ex24_kcist.sbatch
#
# Every result file is written atomically and finished campaigns are skipped, so a killed or timed-out run is
# resumed by submitting the same stage again (python bo_benchmark.py status shows what is missing).
#SBATCH --job-name=ex24
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=_ex24/slurm_%x_%A_%a.out

set -uo pipefail
EX24_STAGE="${EX24_STAGE:?set EX24_STAGE=featurize|encode|calibrate|run}"
REPO="${REPO:-$HOME/Programming/graph_hdc}"
CODE="${CODE:-$HOME/Programming/graph_hdc_ex24}"
VENV="${VENV:-$REPO/.venv}"
FP="$CODE/graph_hdc/experiments/fingerprints"
WORKERS="${WORKERS:-64}"
THREADS="${THREADS:-1}"
GPUS="${GPUS:--1}"   # -1: all GPUs visible to the job (torch.cuda.device_count())
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
CALIBRATE_ARGS="${CALIBRATE_ARGS:-}"
FEATURIZE_ARGS="${FEATURIZE_ARGS:-}"   # e.g. "--libraries beet"

source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
cd "$FP" || { echo "!!! no folder $FP"; exit 2; }
mkdir -p _ex24
# per-job temporary folder on the node-local NVMe disk, removed at the end
export TMPDIR="/tmp/ex24_${SLURM_JOB_ID:-local}_${SHARD}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONWARNINGS=ignore
# many workers share each GPU; expandable segments reduce fragmentation of the shared memory
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== ex24 STAGE=$EX24_STAGE host=$(hostname) shard=$SHARD/$NUM_SHARDS workers=$WORKERS threads=$THREADS gpus=$GPUS $(date) ==="
git -C "$CODE" log -1 --format='code: %h %s' || true
python -c "import graph_hdc, torch, botorch; print('graph_hdc from', graph_hdc.__file__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), '| botorch', botorch.__version__)"
python -c "import os, sys, graph_hdc; sys.exit(not os.path.realpath(graph_hdc.__file__).startswith(os.path.realpath('$CODE') + os.sep))" \
    || { echo "!!! graph_hdc is not imported from CODE=$CODE"; exit 2; }

case "$EX24_STAGE" in
    featurize) python bo_benchmark.py featurize --workers "$WORKERS" $FEATURIZE_ARGS ;;
    encode)    python bo_benchmark.py encode --workers "$WORKERS" --threads "$THREADS" ;;
    calibrate) python bo_benchmark.py calibrate --gpus "$GPUS" $CALIBRATE_ARGS ;;
    run)       python bo_benchmark.py run --workers "$WORKERS" --threads "$THREADS" --gpus "$GPUS" \
                   --shard "$SHARD" --num-shards "$NUM_SHARDS" ;;
    *)         echo "!!! unknown EX24_STAGE=$EX24_STAGE"; exit 2 ;;
esac
RC=$?
echo "=== ex24 STAGE=$EX24_STAGE finished with exit code $RC $(date) ==="
exit $RC
