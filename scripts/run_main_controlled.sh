#!/usr/bin/env bash

set -uo pipefail

ROOT="/c/Users/nsubedi/Documents/lgode/LG-ODE"
DATA="$ROOT/data/simbench_paper/1-MV-rural--0-sw.npz"

EXPERIMENT="protocol_v3_main_extrapolation_obs40_bs8"
RESULTS="$ROOT/results/$EXPERIMENT"
CHECKPOINTS="$ROOT/checkpoints/$EXPERIMENT"
LOGS="$ROOT/logs/$EXPERIMENT"

MODELS=(persistence latentode lgode atode)
SEEDS=(1 2 3)

TASK="extrapolation"
OBSERVED_FRACTION="0.4"
OBS_TAG="40"

mkdir -p "$RESULTS" "$CHECKPOINTS" "$LOGS"

cd "$ROOT" || exit 1

if [[ -f "$ROOT/.venv/Scripts/activate" ]]; then
    source "$ROOT/.venv/Scripts/activate"
fi

if [[ ! -f "$DATA" ]]; then
    echo "ERROR: Dataset not found:"
    echo "$DATA"
    exit 1
fi

echo "======================================================================"
echo "Main controlled LG-ODE/AT-ODE experiment"
echo "Task:                 $TASK"
echo "Observed fraction:    $OBSERVED_FRACTION"
echo "Context length:       12"
echo "Forecast length:      12"
echo "Batch size:           8"
echo "Maximum epochs:       200"
echo "Seeds:                ${SEEDS[*]}"
echo "Models:               ${MODELS[*]}"
echo "Results:              $RESULTS"
echo "======================================================================"

for SEED in "${SEEDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        RUN="${EXPERIMENT}__${TASK}__obs${OBS_TAG}__${MODEL}__seed${SEED}"
        RESULT_FILE="$RESULTS/$RUN.json"
        LOG_FILE="$LOGS/$RUN.log"

        if [[ -f "$RESULT_FILE" ]]; then
            echo
            echo "SKIP completed run: $RUN"
            continue
        fi

        echo
        echo "======================================================================"
        echo "Starting: $RUN"
        echo "Model:    $MODEL"
        echo "Seed:     $SEED"
        echo "Time:     $(date)"
        echo "======================================================================"

        python -u "$ROOT/run_powergrid_lgode.py" \
            --data-path "$DATA" \
            --model "$MODEL" \
            --task "$TASK" \
            --observed-fraction "$OBSERVED_FRACTION" \
            --trajectory-length 24 \
            --context-length 12 \
            --forecast-length 12 \
            --stride 24 \
            --niters 200 \
            --batch-size 8 \
            --lr 5e-4 \
            --weight-decay 0 \
            --gradient-clip 10 \
            --kl-coef 1 \
            --kl-warmup-epochs 10 \
            --train-samples 1 \
            --eval-samples 1 \
            --patience 25 \
            --lr-patience 8 \
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
            --ode-dropout 0.0 \
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

        PIPE_STATUSES=("${PIPESTATUS[@]}")
        PYTHON_STATUS="${PIPE_STATUSES[0]}"
        TEE_STATUS="${PIPE_STATUSES[1]}"

        if [[ ! -f "$RESULT_FILE" ]]; then
            echo
            echo "FAILED: $RUN"
            echo "Python exit status: $PYTHON_STATUS"
            echo "tee exit status:    $TEE_STATUS"
            echo "Result JSON was not written:"
            echo "$RESULT_FILE"
            echo "Last 100 log lines:"
            tail -n 100 "$LOG_FILE"
            exit 1
        fi

        if ! python - "$RESULT_FILE" <<'JSONCHECK'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

if not isinstance(data, dict):
    raise TypeError("Result JSON root must be an object")

if "test" not in data:
    raise KeyError("Result JSON has no test section")

print(f"Validated result JSON: {path}")
JSONCHECK
        then
            echo "ERROR: Invalid result JSON: $RESULT_FILE"
            exit 1
        fi

        if [[ "$PYTHON_STATUS" -ne 0 ]]; then
            echo "WARNING: runner returned $PYTHON_STATUS after writing a valid result."
            echo "Continuing because the result JSON exists and passed validation."
        fi

        if [[ "$TEE_STATUS" -ne 0 ]]; then
            echo "ERROR: tee failed with exit status $TEE_STATUS"
            exit "$TEE_STATUS"
        fi

        if [[ "$MODEL" == "lgode" || "$MODEL" == "atode" ]]; then
            if ! grep -Eq 'physical edges:[[:space:]]+194' "$LOG_FILE"; then
                echo "ERROR: Expected 194 physical edges in $LOG_FILE"
                exit 1
            fi

            if ! grep -Eq 'candidate pairs:[[:space:]]+194' "$LOG_FILE"; then
                echo "ERROR: Expected 194 candidate pairs in $LOG_FILE"
                echo "Do not continue if the model is using 9312 pairs."
                exit 1
            fi
        fi

        if grep -Eiq 'Traceback|AssertionError|RuntimeError|(^|[^a-z])nan([^a-z]|$)' "$LOG_FILE"; then
            echo "ERROR: Possible failure or nonfinite value found in:"
            echo "$LOG_FILE"
            exit 1
        fi

        echo "Completed: $RUN"
    done
done

echo
echo "======================================================================"
echo "All controlled runs completed."
echo "Results: $RESULTS"
echo "======================================================================"

SUMMARY="$RESULTS/summary"
mkdir -p "$SUMMARY"

python -u "$ROOT/scripts/summarize_powergrid_lgode.py" \
    --input-dir "$RESULTS" \
    --output-dir "$SUMMARY" \
    --pattern "${EXPERIMENT}__${TASK}__obs${OBS_TAG}__*__seed*.json"

SUMMARY_STATUS=$?

if [[ "$SUMMARY_STATUS" -ne 0 ]]; then
    echo "WARNING: Runs completed, but summarization failed."
    exit "$SUMMARY_STATUS"
fi

echo
echo "Result JSON count:"
find "$RESULTS" -maxdepth 1 -type f \
    -name "${EXPERIMENT}__${TASK}__obs${OBS_TAG}__*__seed*.json" |
    wc -l

echo
echo "Summary files:"
find "$SUMMARY" -maxdepth 1 -type f -print

echo
echo "Experiment completed successfully."
