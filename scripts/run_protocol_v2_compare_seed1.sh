#!/usr/bin/env bash

set -uo pipefail

ROOT="/c/Users/nsubedi/Documents/lgode/LG-ODE"
DATA="$ROOT/data/simbench_paper/1-MV-rural--0-sw.npz"

RESULTS="$ROOT/results/protocol_v2_compare_seed1"
CHECKPOINTS="$ROOT/checkpoints/protocol_v2_compare_seed1"
LOGS="$ROOT/logs/protocol_v2_compare_seed1"

MODELS=(persistence latentode lgode atode)

TASK="interpolation"
OBSERVED_FRACTION="0.4"
OBS_TAG="40"
SEED="1"

mkdir -p "$RESULTS" "$CHECKPOINTS" "$LOGS"

cd "$ROOT" || exit 1

if [[ -f "$ROOT/.venv/Scripts/activate" ]]; then
    source "$ROOT/.venv/Scripts/activate"
fi

echo "============================================================"
echo "Corrected protocol-v2 four-model comparison"
echo "Task:                 $TASK"
echo "Observed fraction:    $OBSERVED_FRACTION"
echo "Seed:                 $SEED"
echo "Batch size:           4"
echo "Results:              $RESULTS"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    RUN="protocol_v2_compare__${TASK}__obs${OBS_TAG}__${MODEL}__seed${SEED}"
    RESULT_FILE="$RESULTS/$RUN.json"
    LOG_FILE="$LOGS/$RUN.log"

    if [[ -f "$RESULT_FILE" ]]; then
        echo
        echo "SKIP completed run: $RUN"
        continue
    fi

    echo
    echo "============================================================"
    echo "Starting: $RUN"
    echo "Time:     $(date)"
    echo "============================================================"

    python -u "$ROOT/run_powergrid_lgode.py" \
        --data-path "$DATA" \
        --model "$MODEL" \
        --task "$TASK" \
        --observed-fraction "$OBSERVED_FRACTION" \
        --trajectory-length 24 \
        --context-length 12 \
        --forecast-length 12 \
        --stride 24 \
        --niters 50 \
        --batch-size 4 \
        --lr 5e-4 \
        --gradient-clip 10 \
        --kl-warmup-epochs 10 \
        --train-samples 1 \
        --eval-samples 1 \
        --patience 10 \
        --lr-patience 4 \
        --lr-factor 0.5 \
        --latent-dim 16 \
        --recognition-dim 64 \
        --ode-hidden-dim 128 \
        --augmentation-dim 0 \
        --encoder-layers 2 \
        --ode-layers 1 \
        --attention-heads 1 \
        --edge-types 2 \
        --dropout 0.2 \
        --observation-std 0.01 \
        --solver rk4 \
        --rtol 1e-3 \
        --atol 1e-4 \
        --transport-bins 32 \
        --transport-max-age 4 \
        --transport-hidden-dim 64 \
        --transport-attention-dim 16 \
        --transport-heads 4 \
        --transport-speed 1 \
        --transport-decay 1 \
        --seed "$SEED" \
        --mask-seed "$SEED" \
        --eval-seed 12345 \
        --deterministic \
        --num-workers 0 \
        --device auto \
        --results-dir "$RESULTS" \
        --checkpoint-dir "$CHECKPOINTS" \
        --run-name "$RUN" \
        2>&1 | tee "$LOG_FILE"

    PYTHON_STATUS=${PIPESTATUS[0]}

    if [[ "$PYTHON_STATUS" -ne 0 ]]; then
        echo
        echo "FAILED: $RUN"
        echo "Exit status: $PYTHON_STATUS"
        echo "Last 100 log lines:"
        tail -n 100 "$LOG_FILE"
        exit "$PYTHON_STATUS"
    fi

    if [[ ! -f "$RESULT_FILE" ]]; then
        echo "ERROR: command succeeded but result JSON was not found:"
        echo "$RESULT_FILE"
        exit 1
    fi

    echo "Completed: $RUN"
done

echo
echo "============================================================"
echo "All four corrected comparison runs completed."
echo "Results: $RESULTS"
echo "============================================================"

SUMMARY="$RESULTS/summary"
mkdir -p "$SUMMARY"

python -u "$ROOT/scripts/summarize_powergrid_lgode.py" \
    --input-dir "$RESULTS" \
    --output-dir "$SUMMARY" \
    --pattern "protocol_v2_compare__*.json"

SUMMARY_STATUS=$?

if [[ "$SUMMARY_STATUS" -ne 0 ]]; then
    echo "WARNING: training completed, but summarization failed."
    exit "$SUMMARY_STATUS"
fi

echo
echo "Summary files:"
find "$SUMMARY" -maxdepth 1 -type f -print
