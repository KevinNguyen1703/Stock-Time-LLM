#!/bin/bash
# BASELINE: Pure MSE Loss (no directional penalty)
# Use this as baseline to compare with directional loss

echo "============================================"
echo "BASELINE: MSE Loss Only"
echo "============================================"

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2028 run_stock_directional_only.py \
  --loss_type mse \
  --patching_mode single \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/stock/ \
  --data_path vcb_stock_indicators.csv \
  --model_id VCB_stock_baseline \
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
  --model_comment "Baseline-MSE"

