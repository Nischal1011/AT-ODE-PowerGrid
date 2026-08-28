#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/Scripts/python.exe}"
RAW="$ROOT/data/ieee39_transient/raw/tsa_data.pkl"
PROCESSED="$ROOT/data/ieee39_transient/processed/ieee39_transient_v1.npz"
METADATA="$ROOT/data/ieee39_transient/processed/ieee39_transient_v1_metadata.json"
RUN_MODE="${RUN_MODE:-publication}"
OVERWRITE="${OVERWRITE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"

case "$RUN_MODE" in
    smoke)
        FRACTIONS=(0.2 0.8)
        SEEDS=(1)
        EPOCHS=1
        PATIENCE=1
        VALIDATION_INTERVAL=1
        BATCH_SIZE="${BATCH_SIZE:-16}"
        MAX_BATCH_ARGS=(--max-train-batches 2 --max-eval-batches 2)
        ;;
    development)
        FRACTIONS=(0.2 0.8)
        SEEDS=(1)
        EPOCHS=20
        PATIENCE=5
        VALIDATION_INTERVAL=2
        BATCH_SIZE="${BATCH_SIZE:-64}"
        MAX_BATCH_ARGS=()
        ;;
    publication)
        FRACTIONS=(0.2 0.4 0.6 0.8)
        SEEDS=(1 2 3 4 5)
        EPOCHS=150
        PATIENCE=15
        VALIDATION_INTERVAL=2
        BATCH_SIZE="${BATCH_SIZE:-64}"
        MAX_BATCH_ARGS=()
        ;;
    *)
        echo "ERROR: RUN_MODE must be smoke, development, or publication." >&2
        exit 2
        ;;
esac

TASKS=(interpolation extrapolation)
MODELS=(persistence latentode lgode atode)
RESULTS="$ROOT/results/ieee39_transient_${RUN_MODE}"
CHECKPOINTS="$ROOT/checkpoints/ieee39_transient_${RUN_MODE}"
LOGS="$ROOT/logs/ieee39_transient_${RUN_MODE}"
FINGERPRINTS="$CHECKPOINTS/protocol_fingerprints"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 127
fi
if [[ ! -f "$RAW" ]]; then
    echo "ERROR: Missing $RAW" >&2
    echo "Place the manually downloaded Mendeley tsa_data.pkl at that path." >&2
    exit 1
fi

if [[ ! -f "$PROCESSED" || ! -f "$METADATA" ]] || ! \
    "$PYTHON" -c "from pathlib import Path; from lib.ieee39_transient_data import load_ieee39_archive; load_ieee39_archive(Path(r'$PROCESSED'))"; then
    echo "Building or repairing the IEEE39 processed cache."
    "$PYTHON" -m scripts.preprocess_ieee39_transient --force
fi

mkdir -p "$RESULTS" "$CHECKPOINTS" "$LOGS" "$FINGERPRINTS"
"$PYTHON" -m scripts.diagnose_ieee39_transient \
    --data-path "$PROCESSED" --scale "$RUN_MODE" --mask-seed 1 \
    --output "$RESULTS/dataset_diagnostics.json" >/dev/null

echo "IEEE39 transient benchmark"
echo "Run mode: $RUN_MODE"
echo "Tasks: ${TASKS[*]}"
echo "Models: ${MODELS[*]}"
echo "Fractions: ${FRACTIONS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Batch size: $BATCH_SIZE"
echo "Graph: G01-G10 complete directed candidate, 90 edges"

validate_result() {
    "$PYTHON" -m scripts.validate_powergrid_result \
        --result "$1" --data "$PROCESSED" --fingerprint "$2" \
        --model "$3" --task "$4" --seed "$5" \
        --dataset-format ieee39_transient --run-mode "$RUN_MODE" \
        --observed-fraction "$6" --batch-size "$BATCH_SIZE"
}

for TASK in "${TASKS[@]}"; do
    for FRACTION in "${FRACTIONS[@]}"; do
        OBS_TAG=$("$PYTHON" -c "print(int(round(100*float('$FRACTION'))))")
        for SEED in "${SEEDS[@]}"; do
            FINGERPRINT="$FINGERPRINTS/maskv2__${TASK}__obs${OBS_TAG}__seed${SEED}.json"
            for MODEL in "${MODELS[@]}"; do
                RUN="ieee39_transient__${TASK}__obs${OBS_TAG}__${MODEL}__seed${SEED}"
                RESULT_FILE="$RESULTS/$RUN.json"
                LOG_FILE="$LOGS/$RUN.log"
                if [[ -f "$RESULT_FILE" && "$OVERWRITE" != "1" ]]; then
                    validate_result "$RESULT_FILE" "$FINGERPRINT" "$MODEL" \
                        "$TASK" "$SEED" "$FRACTION"
                    continue
                fi
                OVERWRITE_ARG=()
                [[ "$OVERWRITE" == "1" ]] && OVERWRITE_ARG=(--overwrite)
                echo "Starting $RUN"
                set +e
                "$PYTHON" -u run_powergrid_lgode.py \
                    --dataset-format ieee39_transient \
                    --scenario-scale "$RUN_MODE" \
                    --data-path "$PROCESSED" \
                    --model "$MODEL" --task "$TASK" \
                    --observed-fraction "$FRACTION" \
                    --trajectory-length 60 --context-length 30 \
                    --forecast-length 30 --stride 1 \
                    --batch-size "$BATCH_SIZE" --niters "$EPOCHS" \
                    --patience "$PATIENCE" \
                    --validation-interval "$VALIDATION_INTERVAL" \
                    --min-delta 0 --lr 5e-4 --optimizer adam \
                    --lr-patience 5 --lr-factor 0.5 --weight-decay 0 \
                    --gradient-clip 10 --kl-coef 1 --kl-warmup-epochs 10 \
                    --train-samples 1 --eval-samples 1 --latent-dim 16 \
                    --recognition-dim 64 --ode-hidden-dim 128 \
                    --augmentation-dim 0 --encoder-layers 2 --ode-layers 1 \
                    --attention-heads 1 --edge-types 2 \
                    --graph-mode complete_directed_generator_candidate \
                    --dropout 0.2 --ode-dropout 0 --observation-std 0.01 \
                    --solver rk4 --rtol 1e-3 --atol 1e-4 \
                    --transport-bins 32 --transport-max-age 0.6 \
                    --transport-hidden-dim 64 --transport-attention-dim 16 \
                    --transport-heads 4 --transport-speed 1 \
                    --transport-decay 1 --seed "$SEED" --mask-seed "$SEED" \
                    --eval-seed 12345 --deterministic \
                    --num-workers "$NUM_WORKERS" --device "$DEVICE" \
                    --results-dir "$RESULTS" --checkpoint-dir "$CHECKPOINTS" \
                    --protocol-fingerprint-path "$FINGERPRINT" \
                    --run-name "$RUN" "${MAX_BATCH_ARGS[@]}" \
                    "${OVERWRITE_ARG[@]}" 2>&1 | tee "$LOG_FILE"
                PIPE_STATUSES=("${PIPESTATUS[@]}")
                set -e
                echo "Python exit status: ${PIPE_STATUSES[0]}"
                echo "tee exit status:    ${PIPE_STATUSES[1]}"
                [[ "${PIPE_STATUSES[0]}" -eq 0 ]] || exit "${PIPE_STATUSES[0]}"
                [[ "${PIPE_STATUSES[1]}" -eq 0 ]] || exit "${PIPE_STATUSES[1]}"
                if [[ "$MODEL" == "lgode" || "$MODEL" == "atode" ]]; then
                    grep -Eq 'physical edges:[[:space:]]+90' "$LOG_FILE"
                    grep -Eq 'candidate pairs:[[:space:]]+90' "$LOG_FILE"
                fi
                validate_result "$RESULT_FILE" "$FINGERPRINT" "$MODEL" \
                    "$TASK" "$SEED" "$FRACTION"
            done
        done
        SUMMARY="$RESULTS/summary/${TASK}_obs${OBS_TAG}"
        mkdir -p "$SUMMARY"
        set +e
        "$PYTHON" -m scripts.summarize_powergrid_lgode \
            --input-dir "$RESULTS" --output-dir "$SUMMARY" \
            --pattern "ieee39_transient__${TASK}__obs${OBS_TAG}__*__seed*.json" \
            --strict 2>&1 | tee "$SUMMARY/summary.txt"
        SUMMARY_STATUSES=("${PIPESTATUS[@]}")
        set -e
        [[ "${SUMMARY_STATUSES[0]}" -eq 0 && "${SUMMARY_STATUSES[1]}" -eq 0 ]]
    done
done

EXPECTED_RESULTS=$(( ${#TASKS[@]} * ${#FRACTIONS[@]} * ${#SEEDS[@]} * ${#MODELS[@]} ))
ACTUAL_RESULTS=$(find "$RESULTS" -maxdepth 1 -type f \
    -name 'ieee39_transient__*.json' | wc -l)
if [[ "$ACTUAL_RESULTS" -ne "$EXPECTED_RESULTS" ]]; then
    echo "ERROR: Expected $EXPECTED_RESULTS results, found $ACTUAL_RESULTS." >&2
    exit 1
fi

for FRACTION in "${FRACTIONS[@]}"; do
    OBS_TAG=$("$PYTHON" -c "print(int(round(100*float('$FRACTION'))))")
    "$PYTHON" -m scripts.publication_powergrid_report \
        --input-dir "$RESULTS" \
        --output-dir "$RESULTS/publication/obs${OBS_TAG}" \
        --pattern "ieee39_transient__*__obs${OBS_TAG}__*__seed*.json" \
        --seeds "${SEEDS[@]}"
done