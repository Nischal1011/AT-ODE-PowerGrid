#!/usr/bin/env bash
set -uo pipefail

ROOT="/c/Users/nsubedi/Documents/lgode/LG-ODE"
DATA="$ROOT/data/simbench_paper/1-MV-rural--0-sw.npz"
PYTHON="$ROOT/.venv/Scripts/python.exe"
EXPERIMENT="protocol_v5_publication_both_tasks_obs40_bs8"
RESULTS="$ROOT/results/$EXPERIMENT"
CHECKPOINTS="$ROOT/checkpoints/$EXPERIMENT"
LOGS="$ROOT/logs/$EXPERIMENT"
FINGERPRINTS="$CHECKPOINTS/protocol_fingerprints"

TASKS=(interpolation extrapolation)
MODELS=(persistence latentode lgode atode)
SEEDS=(1 2 3 4 5)

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 127
fi
if [[ ! -f "$DATA" ]]; then
    echo "ERROR: Dataset not found: $DATA" >&2
    exit 1
fi

mkdir -p "$RESULTS" "$CHECKPOINTS" "$LOGS" "$FINGERPRINTS"
cd "$ROOT" || exit 1

echo "Experiment: $EXPERIMENT"
echo "Tasks: ${TASKS[*]}"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Primary graph mode: physical_sparse"
echo "Expected graph counts: physical=194 candidates=194"

validate_result() {
    "$PYTHON" "$ROOT/scripts/validate_powergrid_result.py" \
        --result "$1" \
        --data "$DATA" \
        --fingerprint "$2" \
        --model "$3" \
        --task "$4" \
        --seed "$5"
}

for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        FINGERPRINT="$FINGERPRINTS/${TASK}__seed${SEED}.json"
        for MODEL in "${MODELS[@]}"; do
            RUN="${EXPERIMENT}__${TASK}__obs40__${MODEL}__seed${SEED}"
            RESULT_FILE="$RESULTS/$RUN.json"
            LOG_FILE="$LOGS/$RUN.log"

            if [[ -f "$RESULT_FILE" ]]; then
                validate_result "$RESULT_FILE" "$FINGERPRINT" "$MODEL" "$TASK" "$SEED" || exit $?
                continue
            fi

            echo "Starting: $RUN"
            "$PYTHON" -u "$ROOT/run_powergrid_lgode.py" \
                --data-path "$DATA" \
                --model "$MODEL" \
                --task "$TASK" \
                --observed-fraction 0.4 \
                --trajectory-length 24 \
                --context-length 12 \
                --forecast-length 12 \
                --stride 24 \
                --batch-size 8 \
                --niters 200 \
                --patience 25 \
                --min-delta 0 \
                --lr 5e-4 \
                --optimizer adam \
                --lr-patience 8 \
                --lr-factor 0.5 \
                --weight-decay 0 \
                --gradient-clip 10 \
                --kl-coef 1 \
                --kl-warmup-epochs 10 \
                --train-samples 1 \
                --eval-samples 1 \
                --latent-dim 16 \
                --recognition-dim 64 \
                --ode-hidden-dim 128 \
                --augmentation-dim 0 \
                --encoder-layers 2 \
                --ode-layers 1 \
                --attention-heads 1 \
                --edge-types 2 \
                --graph-mode physical_sparse \
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
                --protocol-fingerprint-path "$FINGERPRINT" \
                --run-name "$RUN" \
                2>&1 | tee "$LOG_FILE"
            PIPE_STATUSES=("${PIPESTATUS[@]}")
            PYTHON_STATUS="${PIPE_STATUSES[0]}"
            TEE_STATUS="${PIPE_STATUSES[1]}"
            echo "Python exit status: $PYTHON_STATUS"
            echo "tee exit status:    $TEE_STATUS"
            if [[ "$PYTHON_STATUS" -ne 0 ]]; then
                exit "$PYTHON_STATUS"
            fi
            if [[ "$TEE_STATUS" -ne 0 ]]; then
                exit "$TEE_STATUS"
            fi
            if [[ ! -f "$RESULT_FILE" ]]; then
                echo "ERROR: Missing result JSON: $RESULT_FILE" >&2
                exit 1
            fi
            if [[ "$MODEL" == "lgode" || "$MODEL" == "atode" ]]; then
                grep -Eq 'physical edges:[[:space:]]+194' "$LOG_FILE" || exit 1
                grep -Eq 'candidate pairs:[[:space:]]+194' "$LOG_FILE" || exit 1
            fi
            validate_result "$RESULT_FILE" "$FINGERPRINT" "$MODEL" "$TASK" "$SEED" || exit $?
        done
    done
done

RESULT_COUNT=$(find "$RESULTS" -maxdepth 1 -type f \
    -name "${EXPERIMENT}__*__obs40__*__seed*.json" | wc -l)
if [[ "$RESULT_COUNT" -ne 40 ]]; then
    echo "ERROR: Expected exactly 40 result JSON files, found $RESULT_COUNT" >&2
    exit 1
fi

for TASK in "${TASKS[@]}"; do
    "$PYTHON" "$ROOT/scripts/summarize_powergrid_lgode.py" \
        --input-dir "$RESULTS" \
        --output-dir "$RESULTS/summary/$TASK" \
        --pattern "${EXPERIMENT}__${TASK}__obs40__*__seed*.json" \
        --strict || exit $?
done

"$PYTHON" "$ROOT/scripts/publication_powergrid_report.py" \
    --input-dir "$RESULTS" \
    --output-dir "$RESULTS/publication" \
    --pattern "${EXPERIMENT}__*__obs40__*__seed*.json" \
    --seeds 1 2 3 4 5