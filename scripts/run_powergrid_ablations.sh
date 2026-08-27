#!/usr/bin/env bash
set -uo pipefail

ROOT="/c/Users/nsubedi/Documents/lgode/LG-ODE"
PYTHON="$ROOT/.venv/Scripts/python.exe"
DATA="$ROOT/data/simbench_paper/1-MV-rural--0-sw.npz"
ABLATION="${1:-}"

case "$ABLATION" in
    capacity_matched)
        MODELS=(lgode)
        EXTRA_ARGS=(--recognition-dim 40 --ode-hidden-dim 48)
        ;;
    official_like)
        MODELS=(lgode atode)
        EXTRA_ARGS=(--augmentation-dim 64 --optimizer adamw --weight-decay 1e-3 --train-samples 3)
        ;;
    all_pairs_nri)
        MODELS=(lgode atode)
        EXTRA_ARGS=(--graph-mode all_pairs_nri)
        ;;
    *)
        echo "Usage: bash scripts/run_powergrid_ablations.sh {capacity_matched|official_like|all_pairs_nri}" >&2
        exit 2
        ;;
esac

EXPERIMENT="protocol_v5_ablation_${ABLATION}"
RESULTS="$ROOT/results/$EXPERIMENT"
CHECKPOINTS="$ROOT/checkpoints/$EXPERIMENT"
LOGS="$ROOT/logs/$EXPERIMENT"
mkdir -p "$RESULTS" "$CHECKPOINTS" "$LOGS"
cd "$ROOT" || exit 1

for TASK in interpolation extrapolation; do
    for SEED in 1 2 3 4 5; do
        for MODEL in "${MODELS[@]}"; do
            RUN="${EXPERIMENT}__${TASK}__${MODEL}__seed${SEED}"
            if [[ -f "$RESULTS/$RUN.json" ]]; then
                echo "Refusing to overwrite existing ablation result: $RESULTS/$RUN.json" >&2
                exit 1
            fi
            "$PYTHON" -u run_powergrid_lgode.py \
                --data-path "$DATA" --model "$MODEL" --task "$TASK" \
                --observed-fraction 0.4 --trajectory-length 24 \
                --context-length 12 --forecast-length 12 --stride 24 \
                --batch-size 8 --niters 200 --patience 25 --min-delta 0 \
                --lr 5e-4 --lr-patience 8 --lr-factor 0.5 \
                --optimizer adam --weight-decay 0 --gradient-clip 10 \
                --kl-coef 1 --kl-warmup-epochs 10 --train-samples 1 \
                --eval-samples 1 --latent-dim 16 --recognition-dim 64 \
                --ode-hidden-dim 128 --augmentation-dim 0 \
                --encoder-layers 2 --ode-layers 1 --attention-heads 1 \
                --edge-types 2 --graph-mode physical_sparse --dropout 0.2 \
                --ode-dropout 0 --observation-std 0.01 --solver rk4 \
                --rtol 1e-3 --atol 1e-4 --transport-bins 32 \
                --transport-max-age 4 --transport-hidden-dim 64 \
                --transport-attention-dim 16 --transport-heads 4 \
                --transport-speed 1 --transport-decay 1 \
                --seed "$SEED" --mask-seed "$SEED" --eval-seed 12345 \
                --deterministic --num-workers 0 --device auto \
                --results-dir "$RESULTS" --checkpoint-dir "$CHECKPOINTS" \
                --run-name "$RUN" "${EXTRA_ARGS[@]}" \
                2>&1 | tee "$LOGS/$RUN.log"
            STATUSES=("${PIPESTATUS[@]}")
            echo "Python exit status: ${STATUSES[0]}"
            echo "tee exit status:    ${STATUSES[1]}"
            [[ "${STATUSES[0]}" -eq 0 && "${STATUSES[1]}" -eq 0 ]] || exit 1
        done
    done
done