#!/bin/bash
# Whole-node Booster runner for Experiment 13. Packs the ex_13 command files onto one node with
# GNU parallel: a CPU pool (each task pinned to 1 thread, no GPU) plus, for the table stage, a GPU
# pool that round-robins the 4 GH200s. Drive it with env vars:
#     STAGE=hpo|table   (required)
#     CPU_J=<n>         (parallel CPU slots, default 220)
#     TIME/PARTITION    (SBATCH overrides via --time / -p on the sbatch call)
# Example: sbatch --job-name=ex13hpo --export=ALL,STAGE=hpo run_ex13.sbatch
#SBATCH --job-name=ex13
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --account=aimatchem
#SBATCH --output=/e/project1/aimatchem/teufel1/graph_hdc/graph_hdc/experiments/fingerprints/_ex13/slurm_%x_%j.out
#SBATCH --error=/e/project1/aimatchem/teufel1/graph_hdc/graph_hdc/experiments/fingerprints/_ex13/slurm_%x_%j.out

set -uo pipefail
STAGE="${STAGE:?set STAGE=hpo|table}"
CPU_J="${CPU_J:-220}"
REPO=/e/project1/aimatchem/teufel1/graph_hdc
FP="$REPO/graph_hdc/experiments/fingerprints"
EX13="$FP/_ex13"

jutil env activate -p aimatchem
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
module load parallel 2>/dev/null || true
source "$REPO/.venv/bin/activate"
cd "$FP"

echo "=== ex13 STAGE=$STAGE CPU_J=$CPU_J host=$(hostname) $(date) ==="
command -v parallel >/dev/null || { echo "FATAL: GNU parallel not available"; exit 1; }
python -c "import torch; print('cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())"

run_cpu() {  # $1 = command file : GNU-parallel work queue, single-threaded tasks, no GPU
    [ -s "$1" ] || { echo "(skip: empty/missing $1)"; return 0; }
    echo "--- CPU parallel -j $CPU_J over $(wc -l < "$1") cmds from $(basename "$1") $(date) ---"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
        parallel -j "$CPU_J" --halt never --joblog "$EX13/joblog_$(basename "$1" .txt)_${SLURM_JOB_ID}.log" :::: "$1"
}

run_gpu() {  # $1 = command file : split round-robin across the 4 GH200s, each chunk runs sequentially
    [ -s "$1" ] || { echo "(skip: empty/missing $1)"; return 0; }
    echo "--- GPU pool (4x GH200) over $(wc -l < "$1") cmds from $(basename "$1") $(date) ---"
    for g in 0 1 2 3; do
        awk -v g="$g" 'NR % 4 == g' "$1" > "$EX13/gpu_chunk_${g}_${SLURM_JOB_ID}.sh"
        ( CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
            bash "$EX13/gpu_chunk_${g}_${SLURM_JOB_ID}.sh" ) &
    done
}

if [ "$STAGE" = "hpo" ]; then
    run_cpu "$EX13/warmup_hpo.txt"   # barrier: build shared load__/stats_ caches once
    run_cpu "$EX13/hpo_primer.txt"   # barrier: build each featurization group once (race-free)
    run_cpu "$EX13/hpo_rest.txt"     # all featurization cache hits
elif [ "$STAGE" = "table" ]; then
    run_cpu "$EX13/warmup_table.txt" # barrier: full-data load__/stats_ caches
    run_gpu "$EX13/table_gpu.txt"    # background: COMPAS + QM9 on the 4 GPUs
    run_cpu "$EX13/table_cpu.txt"    # foreground: everything else, CPU-packed
    wait                             # for the backgrounded GPU chunks
else
    echo "unknown STAGE=$STAGE"; exit 2
fi
echo "=== ex13 STAGE=$STAGE DONE $(date) ==="
