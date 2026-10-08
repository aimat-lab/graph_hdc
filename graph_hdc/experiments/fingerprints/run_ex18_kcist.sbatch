#!/bin/bash
# Whole-node runner for Experiment 18 (HDF vs. Morgan, SECFP and MAP4 with the MLP; see _slurm_ex_18.py) on the
# KCIST cluster ("small" partition nodes: 64 CPU, 4x RTX 4090). Same work-queue design as run_ex14_kcist.sbatch
# (NUM_WORKERS worker loops per node, each pinned to one GPU, pulling the next command from a flock-protected
# counter), with three differences for ex_18, which runs without caches:
#
# * The code comes from CODE, a checkout pinned to one commit (e.g. a detached git worktree), so that work on
#   the shared checkout cannot change the code during the runs. The venv of the main checkout is used, with
#   CODE first on PYTHONPATH.
# * Every job first loads the datasets of its shard into the node's /tmp, one after the other: the runs load
#   the datasets themselves (no load cache), and concurrent first downloads of the same file can corrupt it.
# * Every worker has its own TMPDIR (chem_mat_data rewrites its metadata file in the temp folder on every load).
#
# Pass the variables through the submitting environment, NOT with --export=ALL,VAR=... (on KCIST the latter
# makes most nodes fail the job start). Run from the fingerprints folder of CODE:
#   CODE=$HOME/Programming/graph_hdc_ex18 CMD_FILE=_ex18/kcist_smoke.txt sbatch --array=0-0 --time=00:45:00 run_ex18_kcist.sbatch
#   CODE=$HOME/Programming/graph_hdc_ex18 sbatch --array=0-3 run_ex18_kcist.sbatch
# The array MUST be 0-based and contiguous. Failed command lines are collected in _ex18/failed_<queue>.txt and
# can be re-submitted with CMD_FILE pointing to that file (and --array=0-0).
#SBATCH --job-name=ex18
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=_ex18/slurm_%x_%A_%a.out
set -uo pipefail
NUM_GPUS="${NUM_GPUS:-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"          # 3 concurrent runs per RTX 4090
CODE="${CODE:?set CODE to the pinned checkout of graph_hdc}"
VENV="${VENV:-$HOME/Programming/graph_hdc/.venv}"
FP="$CODE/graph_hdc/experiments/fingerprints"
EX18="$FP/_ex18"
RUN="$EX18/run"
TAG="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
CMD_FILE="${CMD_FILE:-$EX18/kcist_commands.txt}"

source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
cd "$FP"
mkdir -p "$RUN"

# node-local NVMe folder for the temporary files of this job, one subfolder per worker; removed at the end
BASE_TMP="/tmp/ex18_$TAG"
mkdir -p "$BASE_TMP"
trap 'rm -rf "$BASE_TMP"' EXIT

# CPU threads per task: split the allocated cores evenly over the concurrent workers.
THREADS=$(( ${SLURM_CPUS_PER_TASK:-64} / NUM_WORKERS ))
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONWARNINGS=ignore
# Three runs share each GPU; expandable segments reduce fragmentation of the shared memory.
export PYTORCH_ALLOC_CONF=expandable_segments:True

# pycomex creates the namespace folder with a non-atomic "if not exists: mkdir", so concurrent first
# runs of a module race; creating them up front removes the race.
mkdir -p results/predict_molecules__fp results/predict_molecules__hdc

echo "=== ex18 host=$(hostname) tag=$TAG workers=$NUM_WORKERS threads=$THREADS cmd_file=$CMD_FILE $(date) ==="
python -c "import graph_hdc, mhfp, torch; print('graph_hdc from', graph_hdc.__file__, '| mhfp', mhfp.__file__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())"

NUM_FAILED=0

shard() {  # $1 = command file -> prints the path of this array task's shard of it
    # SHARD_ID / SHARD_COUNT override the array variables, e.g. to (re)submit single shards as plain jobs.
    local count="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}" id="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
    local out="$RUN/$(basename "$1" .txt)_shard_${id}_of_${count}_$TAG.txt"
    awk -v n="$count" -v t="$id" '(NR - 1) % n == t' "$1" > "$out"
    echo "$out"
}

preload() {  # $1 = command file -> loads every DATASET_NAME it uses into /tmp, one after the other
    local names
    names=$(grep -o -- "--DATASET_NAME=\"'[a-z0-9_]*'\"" "$1" | sed -E "s/.*'([a-z0-9_]+)'.*/\1/" | sort -u | tr '\n' ' ')
    echo "--- pre-loading datasets: $names$(date) ---"
    TMPDIR="$BASE_TMP/preload" python - $names <<'EOF'
import os, sys
os.makedirs(os.environ['TMPDIR'], exist_ok=True)
from chem_mat_data.main import load_graph_dataset
for name in sys.argv[1:]:
    print(f'pre-loaded {name}: {len(load_graph_dataset(name, folder_path="/tmp"))} graphs', flush=True)
EOF
}

run_queue() {  # $1 = command file, $2 = number of workers, $3 = number of GPUs (0 = CPU only)
    local file="$1" nworkers="$2" ngpus="$3"
    [ -s "$file" ] || { echo "(skip: empty/missing $file)"; return 0; }
    local name; name="$(basename "$file" .txt)"
    local q="$RUN/$name"
    mkdir -p "$q/logs"
    echo 0 > "$q/counter"
    : > "$q/joblog"
    local total; total=$(wc -l < "$file")
    echo "--- queue $name: $total cmds, $nworkers workers, $ngpus GPUs $(date) ---"

    local w
    for w in $(seq 0 $((nworkers - 1))); do
        (
            if [ "$ngpus" -gt 0 ]; then gpu=$(( w % ngpus )); else gpu=""; fi
            export TMPDIR="$BASE_TMP/w$w"
            mkdir -p "$TMPDIR"
            while true; do
                i=$(flock "$q/counter.lock" bash -c 'n=$(<"$0"); echo $((n + 1)) > "$0"; echo "$n"' "$q/counter")
                [ "$i" -ge "$total" ] && break
                line=$(sed -n "$((i + 1))p" "$file")
                start=$(date +%s)
                CUDA_VISIBLE_DEVICES="$gpu" bash -c "$line" > "$q/logs/task_$i.log" 2>&1
                rc=$?
                printf '%s\t%s\t%s\t%s\n' "$i" "$rc" "$(( $(date +%s) - start ))" "$w" >> "$q/joblog"
                # keep the logs of successful tasks short; failed ones are kept in full for debugging
                [ "$rc" -eq 0 ] && tail -n 20 "$q/logs/task_$i.log" > "$q/logs/task_$i.tail" && rm "$q/logs/task_$i.log"
            done
        ) &
    done
    wait

    # Collect the command lines that failed so that they can be re-submitted (CMD_FILE=<this file>).
    awk -F'\t' '$2 != 0 {print $1 + 1}' "$q/joblog" | while read -r n; do sed -n "${n}p" "$file"; done > "$EX18/failed_$name.txt"
    local nfail; nfail=$(wc -l < "$EX18/failed_$name.txt")
    local ndone; ndone=$(wc -l < "$q/joblog")
    echo "--- queue $name done: $ndone/$total ran, $nfail failed $(date) ---"
    NUM_FAILED=$(( NUM_FAILED + nfail ))
}

SHARD_FILE="$(shard "$CMD_FILE")"
preload "$SHARD_FILE"
run_queue "$SHARD_FILE" "$NUM_WORKERS" "$NUM_GPUS"
echo "=== ex18 DONE, $NUM_FAILED failed $(date) ==="
[ "$NUM_FAILED" -eq 0 ]
