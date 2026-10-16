#!/bin/bash
# Whole-node Booster runner of Experiment 27 (_slurm_ex_27.py) on JUPITER. Runs the command files of a stage with GNU
# parallel: first the primer and then the warmup file on the CPUs (barriers that write the dataset and HDF encoding
# caches; the job stops if one of their runs fails), then the GPU file and the CPU file side by side.
#   GPU pool: GPU_J slots over the 4 GPUs, slot s on GPU (s - 1) mod 4, so that several small MLPs share a GPU.
#   CPU pool: CPU_J slots with CPU_THREADS threads each and the GPUs hidden.
# Environment:
#     EX27_STAGE=check|fixed|opt|opteval   (required; selects _ex27/<stage>_primer.txt, <stage>_warmup.txt,
#                                           <stage>_gpu.txt and <stage>_cpu.txt; missing files are skipped)
#     PRIMER_FILE / WARMUP_FILE / GPU_FILE / CPU_FILE   (optional overrides, relative to _ex27/)
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
# not "STAGE": loading the Stages/... module sets an environment variable of that name
RUN_STAGE="${EX27_STAGE:?set EX27_STAGE=check|fixed|opt|opteval|envtest}"
GPU_J="${GPU_J:-32}"
CPU_J="${CPU_J:-64}"
CPU_THREADS="${CPU_THREADS:-4}"
MEMFREE="${MEMFREE:-20G}"   # GNU parallel starts a run only while this much RAM is free
CODE=/e/project1/aimatchem/teufel1/graph_hdc_ex27
FP="$CODE/graph_hdc/experiments/fingerprints"
OUT="$FP/_ex27"
PRIMER_FILE="$OUT/${PRIMER_FILE:-${RUN_STAGE}_primer.txt}"
WARMUP_FILE="$OUT/${WARMUP_FILE:-${RUN_STAGE}_warmup.txt}"
GPU_FILE="$OUT/${GPU_FILE:-${RUN_STAGE}_gpu.txt}"
CPU_FILE="$OUT/${CPU_FILE:-${RUN_STAGE}_cpu.txt}"
if [ "$RUN_STAGE" != envtest ] && [ ! -s "$GPU_FILE" ] && [ ! -s "$CPU_FILE" ]; then
    echo "FATAL: no command files for stage $RUN_STAGE ($GPU_FILE, $CPU_FILE)"
    exit 4
fi

jutil env activate -p aimatchem
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
# the compute nodes have no X11 libraries, which RDKit's drawing extension (imported by chem_mat_data) links against
module load X11
module load parallel 2>/dev/null || true
source "$CODE/.venv/bin/activate"
cd "$FP" || exit 2
export PYTHONWARNINGS=ignore
# The compute nodes have no libcairo, which cairosvg loads on import; chem_mat_data imports cairosvg for molecule
# drawings that the experiments never call. A stub module (jupiter_stubs/cairosvg) lets the import succeed.
if ! ldconfig -p 2>/dev/null | grep -q "libcairo.so.2"; then
    export PYTHONPATH="$FP/jupiter_stubs${PYTHONPATH:+:$PYTHONPATH}"
    echo "no libcairo on this node: using the cairosvg stub"
fi
# pycomex decompresses every cache it reads into a temporary file (the HDF encoding of QM9: several GB per run);
# put them on the scratch file system instead of the node's /tmp
export TMPDIR="${SCRATCH:-/tmp}/tmp/ex27_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

# pycomex creates the namespace folder of a module non-atomically; create them up front (see run_ex13.sbatch)
mkdir -p results/predict_molecules__fp results/predict_molecules__hdc

# the compute nodes have no git: the deploy script records the commit
echo "=== ex27 stage=$RUN_STAGE host=$(hostname) commit=$(cat "$OUT/deployed_commit.txt" 2>/dev/null) $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
# import the experiment modules themselves (everything their runs import at startup), before the first run
python check_imports_ex27.py || { echo "FATAL: the experiment modules cannot be imported on this node"; exit 5; }
command -v parallel >/dev/null || { echo "FATAL: GNU parallel not available"; exit 1; }
if [ "$RUN_STAGE" = envtest ]; then
    echo "=== ex27 envtest: environment ok $(date) ==="
    exit 0
fi

# GPU utilization and memory, and the node's free memory, once a minute (for choosing GPU_J)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 60 \
    > "$OUT/gpu_util_${SLURM_JOB_ID}.csv" 2>/dev/null &
monitor_pids="$!"
( while true; do echo "$(date +%T) $(free -g | awk '/Mem:/ {print "used_gb", $3, "available_gb", $7}')"; sleep 60; done ) \
    > "$OUT/memory_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pids="$monitor_pids $!"

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

for barrier in "$PRIMER_FILE" "$WARMUP_FILE"; do
    run_cpu "$barrier"
    failed=$?
    if [ "$failed" -ne 0 ]; then
        echo "FATAL: $failed run(s) of $(basename "$barrier") failed; delete their (possibly truncated) cache files" \
             "(hdc_*, load__*, stats_*) before a new attempt"
        # queued follow-up rounds of this chain (same job name) would only fail on the same cache files: cancel them
        scancel --user="$USER" --state=PENDING --name="$SLURM_JOB_NAME" \
            && echo "cancelled the pending jobs named $SLURM_JOB_NAME"
        kill $monitor_pids 2>/dev/null
        rm -rf "$TMPDIR"
        exit 3
    fi
done
run_gpu "$GPU_FILE" &
gpu_pid=$!
run_cpu "$CPU_FILE"
echo "CPU pool: $? failed run(s) $(date)"
wait "$gpu_pid"
echo "GPU pool: $? failed run(s) $(date)"
kill $monitor_pids 2>/dev/null
rm -rf "$TMPDIR"
echo "=== ex27 stage=$RUN_STAGE DONE $(date) ==="
