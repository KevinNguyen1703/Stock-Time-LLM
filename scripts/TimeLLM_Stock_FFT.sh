#!/bin/bash
# TimeLLM Stock with FFT+Attention Patching
# Usage: bash scripts/TimeLLM_Stock_FFT.sh [patching_mode]
# patching_mode: frequency_aware (default), multi_scale, single

patching_mode=${1:-frequency_aware}

echo "Running with patching_mode: $patching_mode"

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2026 run_stock_fft.py \
  --patching_mode $patching_mode \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/stock/ \
  --data_path vcb_stock_indicators.csv \
  --model_id VCB_stock_60_1 \
  --model TimeLLM \
  --data Stock \
  --features MS \
  --target "Adj Close" \
  --freq d \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 1 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 6 \
  --train_epochs 30 \
  --patience 10 \
  --patch_len 8 \
  --stride 4 \
  --prompt_domain 1 \
  --model_comment "TimeLLM-$patching_mode"

