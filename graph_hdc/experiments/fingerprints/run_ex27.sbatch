#!/bin/bash
# Whole-node Booster runner of Experiment 27 (_slurm_ex_27.py) on JUPITER. Runs the command files of a stage with GNU
# parallel: first the warmup file on the CPUs (a barrier that writes the HDF encoding caches; the job stops if a warmup
# run fails), then the GPU file and the CPU file side by side.
#   GPU pool: GPU_J slots over the 4 GPUs, slot s on GPU (s - 1) mod 4, so that several small MLPs share a GPU.
#   CPU pool: CPU_J slots with CPU_THREADS threads each and the GPUs hidden.
# Environment:
#     EX27_STAGE=check|fixed      (required; selects _ex27/<stage>_warmup.txt, <stage>_gpu.txt, <stage>_cpu.txt)
#     WARMUP_FILE / GPU_FILE / CPU_FILE   (optional overrides, relative to _ex27/)
#     GPU_J (default 32), CPU_J (default 64), CPU_THREADS (default 4), MEMFREE (default 20G)
# Example (from the fingerprints folder of the ex_27 worktree, after python _slurm_ex_27.py):
#     sbatch --export=ALL,EX27_STAGE=check --time=04:00:00 run_ex27.sbatch
#SBATCH --job-name=ex27
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --account=aimatchem
#SBATCH --output=/e/project1/aimatchem/teufel1/graph_hdc_ex27/graph_hdc/experiments/fingerprints/_ex27/slurm_%x_%j.out
#SBATCH --error=/e/project1/aimatchem/teufel1/graph_hdc_ex27/graph_hdc/experiments/fingerprints/_ex27/slurm_%x_%j.out

set -uo pipefail
STAGE="${EX27_STAGE:?set EX27_STAGE=check|fixed}"
GPU_J="${GPU_J:-32}"
CPU_J="${CPU_J:-64}"
CPU_THREADS="${CPU_THREADS:-4}"
MEMFREE="${MEMFREE:-20G}"   # GNU parallel starts a run only while this much RAM is free
CODE=/e/project1/aimatchem/teufel1/graph_hdc_ex27
FP="$CODE/graph_hdc/experiments/fingerprints"
OUT="$FP/_ex27"
WARMUP_FILE="$OUT/${WARMUP_FILE:-${STAGE}_warmup.txt}"
GPU_FILE="$OUT/${GPU_FILE:-${STAGE}_gpu.txt}"
CPU_FILE="$OUT/${CPU_FILE:-${STAGE}_cpu.txt}"

jutil env activate -p aimatchem
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
module load parallel 2>/dev/null || true
source "$CODE/.venv/bin/activate"
cd "$FP" || exit 2
export PYTHONWARNINGS=ignore

# pycomex creates the namespace folder of a module non-atomically; create them up front (see run_ex13.sbatch)
mkdir -p results/predict_molecules__fp results/predict_molecules__hdc

echo "=== ex27 stage=$STAGE host=$(hostname) commit=$(git -C "$CODE" log -1 --format=%h) $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
command -v parallel >/dev/null || { echo "FATAL: GNU parallel not available"; exit 1; }

run_cpu() {  # $1 = command file; returns the number of failed runs (0 = all succeeded)
    [ -s "$1" ] || { echo "(skip: empty or missing $1)"; return 0; }
    echo "--- CPU pool -j $CPU_J x $CPU_THREADS threads: $(wc -l < "$1") runs from $(basename "$1") $(date) ---"
    OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
        CUDA_VISIBLE_DEVICES="" \
        parallel -j "$CPU_J" --memfree "$MEMFREE" --halt never \
        --joblog "$OUT/joblog_$(basename "$1" .txt)_${SLURM_JOB_ID}.log" :::: "$1"
}

run_gpu() {  # $1 = command file; slot s runs on GPU (s - 1) mod 4
    [ -s "$1" ] || { echo "(skip: empty or missing $1)"; return 0; }
    echo "--- GPU pool -j $GPU_J over 4 GPUs: $(wc -l < "$1") runs from $(basename "$1") $(date) ---"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        parallel -j "$GPU_J" --memfree "$MEMFREE" --halt never \
        --joblog "$OUT/joblog_$(basename "$1" .txt)_${SLURM_JOB_ID}.log" \
        'CUDA_VISIBLE_DEVICES=$(( ({%} - 1) % 4 )) bash -c {}' :::: "$1"
}

run_cpu "$WARMUP_FILE"
failed=$?
if [ "$failed" -ne 0 ]; then
    echo "FATAL: $failed warmup run(s) failed; delete their (possibly truncated) hdc_* cache files before a new attempt"
    exit 3
fi
run_gpu "$GPU_FILE" &
gpu_pid=$!
run_cpu "$CPU_FILE"
echo "CPU pool: $? failed run(s) $(date)"
wait "$gpu_pid"
echo "GPU pool: $? failed run(s) $(date)"
echo "=== ex27 stage=$STAGE DONE $(date) ==="
