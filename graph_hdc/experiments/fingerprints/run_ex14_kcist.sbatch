#!/bin/bash
# Whole-node runner for Experiment 14 (HDF vs. GNNs) on the KCIST cluster ("small" partition nodes:
# 64 CPU, 4x RTX 4090). Executes the command files written by _slurm_ex_14.py with a plain-bash work
# queue (GNU parallel is not installed on KCIST): NUM_WORKERS worker loops per node, each pinned to
# one GPU (round-robin) and pulling the next command from a flock-protected counter, so the long QM9
# runs are load-balanced dynamically.
#
# Stages (EX14_STAGE, required):
#   warmup  build the shared load__/stats_ caches (CPU only). Must finish before anything else.
#   hdf     all HDF runs: the "primer" block first (one run per DATASET_NAME x SEED, builds the HDF
#           encoding caches race-free), then the remaining targets that share those caches.
#   gnn     trained + random GNN runs. Submit as a job array; array task i runs every line with
#           (line_number - 1) % SLURM_ARRAY_TASK_COUNT == i. The array MUST be 0-based and contiguous
#           (--array=0-7, optionally with a %K throttle); --array=1-8 would silently skip shard 0.
#   smoke   warmup + hdf + gnn on the smoke_* command files, on one node.
#   retry   re-run the command lines in RETRY_FILE.
#
# Examples (from the fingerprints folder):
#   sbatch --job-name=ex14warm  --export=ALL,EX14_STAGE=warmup run_ex14_kcist.sbatch
#   sbatch --job-name=ex14hdf   --dependency=afterok:<warm> --export=ALL,EX14_STAGE=hdf run_ex14_kcist.sbatch
#   sbatch --job-name=ex14gnn   --dependency=afterok:<warm> --array=0-7 --export=ALL,EX14_STAGE=gnn run_ex14_kcist.sbatch
#
# Every stage writes a joblog (index, exit code, seconds, worker) and a failed_<...>.txt with the
# command lines that exited non-zero, which can be fed back via EX14_STAGE=retry RETRY_FILE=...
#
# Resume after a timeout / kill / failures (no ex_14 job may be running): on the login node run
#   python _slurm_ex_14.py missing
# which writes the lines without a completed archive to _ex14/missing_{hdf,gnn}.txt, then re-submit
#   ... --export=ALL,EX14_STAGE=hdf,HDF_FILE=_ex14/missing_hdf.txt,HDF_PRIMER_FILE=_ex14/missing_hdf_primer_count.txt
#   ... --array=0-K --export=ALL,EX14_STAGE=gnn,GNN_FILE=_ex14/missing_gnn.txt
#
# The warmup is short CPU work; submit it with e.g. --gres=gpu:0 --cpus-per-task=16 --mem=64G --time=02:00:00.
#SBATCH --job-name=ex14
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=_ex14/slurm_%x_%A_%a.out

set -uo pipefail
EX14_STAGE="${EX14_STAGE:?set EX14_STAGE=warmup|hdf|gnn|smoke|retry}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"          # 3 concurrent runs per RTX 4090
REPO="${REPO:-$HOME/Programming/graph_hdc}"
FP="$REPO/graph_hdc/experiments/fingerprints"
EX14="$FP/_ex14"
RUN="$EX14/run"
TAG="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
GNN_FILE="${GNN_FILE:-$EX14/gnn.txt}"
HDF_FILE="${HDF_FILE:-$EX14/hdf.txt}"
HDF_PRIMER_FILE="${HDF_PRIMER_FILE:-$EX14/hdf_primer_count.txt}"

source "$REPO/.venv/bin/activate"
cd "$FP"
mkdir -p "$RUN"

# pycomex decompresses cache files into a temporary file; the QM9 HDF caches are several GB each and
# up to 12 runs load them at once, so keep these temp files on the large shared /home instead of the
# node-local /tmp (and remove them at the end, also after a failed run).
export TMPDIR="$RUN/tmp_$TAG"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# CPU threads per task: split the allocated cores evenly over the concurrent workers.
THREADS=$(( ${SLURM_CPUS_PER_TASK:-64} / NUM_WORKERS ))
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONWARNINGS=ignore
# Three runs share each GPU; expandable segments reduce fragmentation of the shared memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pycomex creates the namespace folder with a non-atomic "if not exists: mkdir", so concurrent first
# runs of a module race; creating them up front removes the race.
mkdir -p results/predict_molecules__gnn results/predict_molecules__gnn_random results/predict_molecules__hdc

echo "=== ex14 STAGE=$EX14_STAGE host=$(hostname) tag=$TAG workers=$NUM_WORKERS threads=$THREADS $(date) ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())"

NUM_FAILED=0

run_queue() {  # $1 = command file, $2 = number of workers, $3 = number of GPUs (0 = CPU only)
    local file="$1" nworkers="$2" ngpus="$3"
    [ -s "$file" ] || { echo "(skip: empty/missing $file)"; return 0; }
    local name; name="$(basename "$file" .txt)_$TAG"
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

    # Collect the command lines that failed so they can be re-submitted with EX14_STAGE=retry.
    awk -F'\t' '$2 != 0 {print $1 + 1}' "$q/joblog" | while read -r n; do sed -n "${n}p" "$file"; done > "$EX14/failed_$name.txt"
    local nfail; nfail=$(wc -l < "$EX14/failed_$name.txt")
    local ndone; ndone=$(wc -l < "$q/joblog")
    echo "--- queue $name done: $ndone/$total ran, $nfail failed $(date) ---"
    NUM_FAILED=$(( NUM_FAILED + nfail ))
}

shard() {  # $1 = command file -> prints the path of this array task's shard of it
    local count="${SLURM_ARRAY_TASK_COUNT:-1}" id="${SLURM_ARRAY_TASK_ID:-0}"
    local out="$RUN/$(basename "$1" .txt)_shard_${id}_of_${count}.txt"
    awk -v n="$count" -v t="$id" '(NR - 1) % n == t' "$1" > "$out"
    echo "$out"
}

run_hdf() {  # $1 = hdf command file, $2 = file holding the primer line count
    local primer; primer=$(cat "$2")
    head -n "$primer" "$1" > "$RUN/$(basename "$1" .txt)_primer.txt"
    tail -n +"$((primer + 1))" "$1" > "$RUN/$(basename "$1" .txt)_rest.txt"
    local failed_before=$NUM_FAILED
    run_queue "$RUN/$(basename "$1" .txt)_primer.txt" "$NUM_WORKERS" "$NUM_GPUS"
    if [ "$NUM_FAILED" -gt "$failed_before" ]; then
        # A failed primer may leave a missing or truncated encoding cache which the rest runs of the same
        # dataset would then all race to rebuild (or fail to load). Stop here; resume via the missing mode.
        local nrest; nrest=$(wc -l < "$RUN/$(basename "$1" .txt)_rest.txt")
        echo "!!! primer had failures: skipping the $nrest rest runs (resume with: python _slurm_ex_14.py missing)"
        NUM_FAILED=$(( NUM_FAILED + nrest ))
        return
    fi
    run_queue "$RUN/$(basename "$1" .txt)_rest.txt" "$NUM_WORKERS" "$NUM_GPUS"
}

case "$EX14_STAGE" in
    warmup)
        run_queue "$EX14/warmup.txt" 8 0 ;;
    hdf)
        run_hdf "$HDF_FILE" "$HDF_PRIMER_FILE" ;;
    gnn)
        run_queue "$(shard "$GNN_FILE")" "$NUM_WORKERS" "$NUM_GPUS" ;;
    smoke)
        run_queue "$EX14/smoke_warmup.txt" 8 0
        run_hdf "$EX14/smoke_hdf.txt" "$EX14/smoke_hdf_primer_count.txt"
        run_queue "$EX14/smoke_gnn.txt" "$NUM_WORKERS" "$NUM_GPUS" ;;
    retry)
        run_queue "${RETRY_FILE:?set RETRY_FILE}" "$NUM_WORKERS" "$NUM_GPUS" ;;
    *)
        echo "unknown EX14_STAGE=$EX14_STAGE"; exit 2 ;;
esac
echo "=== ex14 STAGE=$EX14_STAGE DONE, $NUM_FAILED failed $(date) ==="
# A non-zero exit makes "--dependency=afterok" hold back the dependent stages (e.g. after a failed
# warmup nothing else should start on half-built caches).
[ "$NUM_FAILED" -eq 0 ]
