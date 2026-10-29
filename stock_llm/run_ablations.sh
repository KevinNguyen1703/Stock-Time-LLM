#!/bin/bash
# Time-LLM+ ablation grid (frozen backbone). Each row isolates one contribution; the other axes stay at
# the setting named in the comment. Results land in reports/plus/*.json (tag encodes every axis).
#   BACKBONE=qwen1.5b FOLDS="0 1 2" SEEDS="0 1 2" bash stock_llm/run_ablations.sh [c1|c2|c3|full|all]
# Rough cost on one RTX 4090 with qwen1.5b: ~20 min per run (3 epochs) -> 15 configs x 3 folds = ~15 h per seed.
cd "$(dirname "$0")/.."
BACKBONE=${BACKBONE:-qwen1.5b}; FOLDS=${FOLDS:-"0 1 2"}; SEEDS=${SEEDS:-"0"}; H=${H:-30}; WHICH=${1:-all}
run() {
  for f in $FOLDS; do for s in $SEEDS; do
    .venv/bin/python -m stock_llm.train_plus --backbone "$BACKBONE" --fold "$f" --seed "$s" --horizon "$H" "$@" \
      > "logs/plus_$(echo "$@" | tr ' ' '_' | tr -d '-')_f${f}_s${s}.log" 2>&1
  done; done
}
if [[ $WHICH == c1 || $WHICH == all ]]; then     # C1 prompting (patching=fixed, loss=mse)
  run --prompt none          --knowledge 0 --patching fixed --loss mse
  run --prompt static        --knowledge 0 --patching fixed --loss mse   # original Time-LLM = baseline
  run --prompt data          --knowledge 0 --patching fixed --loss mse   # earlier dynamic prompt
  run --prompt news          --knowledge 0 --patching fixed --loss mse
  run --prompt news          --knowledge 1 --patching fixed --loss mse
  run --prompt news_shuffled --knowledge 1 --patching fixed --loss mse   # control
fi
if [[ $WHICH == c2 || $WHICH == all ]]; then     # C2 patching (prompt=news + knowledge, loss=mse)
  run --prompt news          --knowledge 1 --patching fft      --loss mse
  run --prompt news          --knowledge 1 --patching fft_attn --loss mse   # earlier FFT+attention
  run --prompt news          --knowledge 1 --patching fft_text --loss mse
  run --prompt news_shuffled --knowledge 1 --patching fft_text --loss mse   # control
fi
if [[ $WHICH == c3 || $WHICH == all ]]; then     # C3 loss (prompt=news + knowledge, patching=fft_text)
  run --prompt news --knowledge 1 --patching fft_text --loss hybrid      # earlier hybrid loss
  run --prompt news --knowledge 1 --patching fft_text --loss decomp
  run --prompt news --knowledge 1 --patching fft_text --loss decomp_dir
  run --prompt news --knowledge 1 --patching fft_text --loss selective   # full Time-LLM+
fi
if [[ $WHICH == full || $WHICH == all ]]; then   # full-model control
  run --prompt news_shuffled --knowledge 1 --patching fft_text --loss selective
fi
