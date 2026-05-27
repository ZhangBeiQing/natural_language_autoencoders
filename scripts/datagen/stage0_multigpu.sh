#!/bin/bash
# Multi-GPU wrapper for stage0 activation extraction.
#
# Launches N parallel stage0 processes (one per GPU, CUDA_VISIBLE_DEVICES=i),
# each processing a disjoint slice of the corpus. Stage0's per-doc keyed RNG
# guarantees the merged output is row-for-row identical to a serial run.
#
# Usage: same args as stage0_extract, plus optional NGPU env var override.
#
#   scripts/datagen/stage0_multigpu.sh \
#       --base-model Qwen/Qwen2.5-7B-Instruct \
#       --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
#       --corpus-length 100000 --positions-per-doc 10 \
#       --layer-index 20 \
#       --output /tmp/base.parquet
#
# Shards are written to ${OUTPUT}.shards/shard_{i}.parquet and left in place
# after merging. Completed shards (valid parquet footer + sidecar) are SKIPPED
# on re-run — so a retry after crash resumes instead of clobbering the 45min
# of GPU work. Gemma-12b 2026-03-10: recovery script re-invoked stage0 after
# merge raced a shard flush; second launch truncated all 8 completed shards.
set -euo pipefail

# --- detect GPUs -------------------------------------------------------------
if [[ -z "${NGPU:-}" ]]; then
    NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
if [[ "$NGPU" -lt 1 ]]; then
    echo "error: NGPU=$NGPU (no GPUs detected and no override set)" >&2
    exit 1
fi

# --- parse args --------------------------------------------------------------
# Capture args we need to slice/redirect; passthrough everything else to stage0.
OUTPUT=""
CORPUS_LENGTH=""
CORPUS_START=0
CORPUS=""
CORPUS_CONFIG=""
CORPUS_SPLIT="train"
STORAGE_ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT="$2"; shift 2 ;;
        --corpus-length)
            CORPUS_LENGTH="$2"; shift 2 ;;
        --corpus-start)
            CORPUS_START="$2"; shift 2 ;;
        --corpus)
            CORPUS="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --corpus-config)
            CORPUS_CONFIG="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --corpus-split)
            CORPUS_SPLIT="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --storage-cls|--storage-kwargs)
            STORAGE_ARGS+=("$1" "$2"); shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    echo "error: --output is required" >&2; exit 1
fi
if [[ -z "$CORPUS_LENGTH" ]]; then
    echo "error: --corpus-length is required" >&2; exit 1
fi
if [[ -z "$CORPUS" ]]; then
    echo "error: --corpus is required" >&2; exit 1
fi
if [[ "$CORPUS_LENGTH" -lt "$NGPU" ]]; then
    echo "error: --corpus-length ($CORPUS_LENGTH) < NGPU ($NGPU)" >&2; exit 1
fi

SHARD_DIR="${OUTPUT}.shards"
mkdir -p "$SHARD_DIR"

# --- compute slices ----------------------------------------------------------
PER_SHARD=$(( CORPUS_LENGTH / NGPU ))

declare -a SHARD_START SHARD_LEN SHARD_OUT
for (( i=0; i<NGPU; i++ )); do
    SHARD_START[i]=$(( CORPUS_START + i * PER_SHARD ))
    if [[ $i -eq $(( NGPU - 1 )) ]]; then
        SHARD_LEN[i]=$(( CORPUS_LENGTH - i * PER_SHARD ))
    else
        SHARD_LEN[i]=$PER_SHARD
    fi
    SHARD_OUT[i]="$SHARD_DIR/shard_${i}.parquet"
done

echo "=== stage0 multi-GPU: $NGPU shards, corpus [$CORPUS_START, $((CORPUS_START + CORPUS_LENGTH))) ==="
for (( i=0; i<NGPU; i++ )); do
    echo "  gpu $i: start=${SHARD_START[i]} len=${SHARD_LEN[i]} → ${SHARD_OUT[i]}"
done
echo

# Populate the HuggingFace datasets cache once before launching GPU workers.
# Without this, every shard process calls load_dataset() at the same time,
# which is both noisy and fragile for remote parquet datasets such as FineWeb.
if [[ "${STAGE0_PRECACHE:-1}" != "0" && ! -f "$CORPUS" ]]; then
    echo "=== pre-caching corpus slice before GPU launch ==="
    echo "  corpus=$CORPUS config=${CORPUS_CONFIG:-<default>} split=${CORPUS_SPLIT}[${CORPUS_START}:$((CORPUS_START + CORPUS_LENGTH))]"
    python - "$CORPUS" "$CORPUS_CONFIG" "$CORPUS_SPLIT" "$CORPUS_START" "$CORPUS_LENGTH" <<'PY'
import sys
from datasets import Dataset, load_dataset

corpus, corpus_config, split, start, length = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
name = corpus_config or None
split_expr = f"{split}[{start}:{start + length}]"
print(f"[precache] load_dataset({corpus!r}, name={name!r}, split={split_expr!r})", flush=True)
ds = load_dataset(corpus, name=name, split=split_expr)
assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds).__name__}"
print(f"[precache] ready rows={len(ds)} columns={ds.column_names}", flush=True)
PY
    echo
elif [[ -f "$CORPUS" ]]; then
    echo "=== local corpus parquet detected; skipping HuggingFace pre-cache ==="
    echo "  corpus=$CORPUS"
    echo
fi

# Shard is done iff its parquet has a valid footer AND its sidecar exists
# (sidecar write is the last thing stage0_extract does). Checking the sidecar
# alone isn't enough: the Gemma-12b race showed you can have a sidecar from
# run 1 alongside a truncated parquet from run 2's abort.
_shard_complete () {
    local pq="$1"
    [[ -f "${pq}.nla_meta.yaml" ]] && python -c "
import sys, pyarrow.parquet as pq
pq.ParquetFile(sys.argv[1]).metadata.num_rows
" "$pq" 2>/dev/null
}

# --- launch ------------------------------------------------------------------
declare -a PIDS SKIPPED
for (( i=0; i<NGPU; i++ )); do
    LOG="$SHARD_DIR/shard_${i}.log"
    if _shard_complete "${SHARD_OUT[i]}"; then
        echo "shard $i: SKIP (valid footer + sidecar at ${SHARD_OUT[i]})"
        SKIPPED[i]=1
        continue
    fi
    SKIPPED[i]=0
    CUDA_VISIBLE_DEVICES=$i python -m nla.datagen.stage0_extract \
        "${EXTRA_ARGS[@]}" \
        ${STORAGE_ARGS[@]+"${STORAGE_ARGS[@]}"} \
        --corpus-start "${SHARD_START[i]}" \
        --corpus-length "${SHARD_LEN[i]}" \
        --output "${SHARD_OUT[i]}" \
        > "$LOG" 2>&1 &
    PIDS[i]=$!
    echo "launched shard $i (pid ${PIDS[i]}) → log: $LOG"
done
echo

MONITOR_INTERVAL="${STAGE0_MONITOR_INTERVAL:-60}"

_pid_active () {
    local pid="$1"
    local stat
    stat="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ -n "$stat" && "$stat" != Z* ]]
}

_latest_log_line () {
    local log="$1"
    if [[ ! -f "$log" ]]; then
        echo "(log not created yet)"
        return
    fi
    tr '\r' '\n' < "$log" | sed '/^[[:space:]]*$/d' | tail -n 1
}

_monitor_shards () {
    local any_running=1
    local round=0
    while [[ "$any_running" -eq 1 ]]; do
        if [[ "$round" -gt 0 ]]; then
            sleep "$MONITOR_INTERVAL"
        fi
        any_running=0
        echo "=== stage0 monitor $(date '+%Y-%m-%d %H:%M:%S') ==="
        for (( i=0; i<NGPU; i++ )); do
            LOG="$SHARD_DIR/shard_${i}.log"
            if [[ "${SKIPPED[i]}" == "1" ]]; then
                echo "  shard $i: SKIPPED"
                continue
            fi
            if _pid_active "${PIDS[i]}"; then
                any_running=1
                STATUS="RUNNING pid=${PIDS[i]}"
            elif kill -0 "${PIDS[i]}" 2>/dev/null; then
                STATUS="EXITED waiting-for-status pid=${PIDS[i]}"
            else
                STATUS="EXITED pid=${PIDS[i]}"
            fi
            SIZE="no-parquet"
            if [[ -f "${SHARD_OUT[i]}" ]]; then
                SIZE="$(du -h "${SHARD_OUT[i]}" | awk '{print $1}')"
            fi
            SIDECAR="no-sidecar"
            if [[ -f "${SHARD_OUT[i]}.nla_meta.yaml" ]]; then
                SIDECAR="sidecar=yes"
            fi
            echo "  shard $i: $STATUS start=${SHARD_START[i]} len=${SHARD_LEN[i]} $SIZE $SIDECAR"
            echo "    latest: $(_latest_log_line "$LOG")"
        done
        echo
        round=$(( round + 1 ))
    done
}

if [[ "$MONITOR_INTERVAL" != "0" ]]; then
    _monitor_shards
fi

# --- wait + check ------------------------------------------------------------
FAILED=0
for (( i=0; i<NGPU; i++ )); do
    if [[ "${SKIPPED[i]}" == "1" ]]; then
        continue
    fi
    if wait "${PIDS[i]}"; then
        echo "shard $i: OK"
    else
        FAILED=1
        echo "shard $i: FAILED (exit $?) — log tail:" >&2
        tail -n 50 "$SHARD_DIR/shard_${i}.log" >&2
        echo >&2
    fi
done

if [[ $FAILED -ne 0 ]]; then
    echo "=== one or more shards failed; shards left in $SHARD_DIR ===" >&2
    exit 1
fi

# --- merge -------------------------------------------------------------------
echo
echo "=== merging shards → $OUTPUT ==="
python -m nla.datagen.merge_base \
    --inputs "${SHARD_OUT[@]}" \
    --output "$OUTPUT" \
    ${STORAGE_ARGS[@]+"${STORAGE_ARGS[@]}"}

echo
echo "=== done; shards kept in $SHARD_DIR ==="
