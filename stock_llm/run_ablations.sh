#!/bin/bash
# Time-LLM+ ablation grid (frozen backbone). Each block isolates one contribution; the other axes stay at
# the setting in the comment. Results: reports/plus/*.json (the tag encodes every axis).
#   BACKBONE=qwen1.5b FOLDS="0 1 2" SEEDS="0" bash stock_llm/run_ablations.sh [c2|c3|c1|all]
# Cost on one RTX 4090 with qwen1.5b: ~20 min per run -> 17 configs x 3 folds ~ 17 h per seed.
cd "$(dirname "$0")/.."
BACKBONE=${BACKBONE:-qwen1.5b}; FOLDS=${FOLDS:-"0 1 2"}; SEEDS=${SEEDS:-"0"}; H=${H:-30}; WHICH=${1:-all}
run() {
  for f in $FOLDS; do for s in $SEEDS; do
    tag="$(echo "$@" | tr ' ' '_' | tr -d '-')_f${f}_s${s}"
    out="reports/plus/${BACKBONE}_f${f}_h${H}_p-$2_k$4_$6_$8_s${s}.json"
    [[ -f $out ]] && continue                      # resumable: skip finished runs
    .venv/bin/python -m stock_llm.train_plus --backbone "$BACKBONE" --fold "$f" --seed "$s" --horizon "$H" "$@" \
      > "logs/plus_${tag}.log" 2>&1
  done; done
}
if [[ $WHICH == c2 || $WHICH == all ]]; then     # C2 spectral (prompt=news + knowledge, loss=mse)
  run --prompt news          --knowledge 1 --patching fixed             --loss mse
  run --prompt news          --knowledge 1 --patching fft_attn          --loss mse   # earlier FFT+attention
  run --prompt news          --knowledge 1 --patching bank              --loss mse
  run --prompt news          --knowledge 1 --patching bank_text         --loss mse
  run --prompt news          --knowledge 1 --patching bank_text_aligned --loss mse   # full KASD
  run --prompt news_shuffled --knowledge 1 --patching bank_text_aligned --loss mse   # control
fi
if [[ $WHICH == c3 || $WHICH == all ]]; then     # C3 objective (prompt=news + knowledge, patching=KASD)
  run --prompt news          --knowledge 1 --patching bank_text_aligned --loss hybrid  # earlier hybrid loss
  run --prompt news          --knowledge 1 --patching bank_text_aligned --loss decomp
  run --prompt news          --knowledge 1 --patching bank_text_aligned --loss dro
  run --prompt news          --knowledge 1 --patching bank_text_aligned --loss dro_rank
  run --prompt news          --knowledge 1 --patching bank_text_aligned --loss full    # full Time-LLM+
  run --prompt news_shuffled --knowledge 1 --patching bank_text_aligned --loss full    # control
fi
if [[ $WHICH == c1 || $WHICH == all ]]; then     # C1 prompting (patching=fixed, loss=mse)
  run --prompt none          --knowledge 0 --patching fixed --loss mse
  run --prompt static        --knowledge 0 --patching fixed --loss mse   # original Time-LLM
  run --prompt data          --knowledge 0 --patching fixed --loss mse   # earlier dynamic prompt
  run --prompt news          --knowledge 0 --patching fixed --loss mse
  run --prompt news_shuffled --knowledge 1 --patching fixed --loss mse   # control
fi
