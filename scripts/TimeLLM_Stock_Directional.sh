#!/bin/bash
# TimeLLM Stock - Directional Loss
# Usage: bash scripts/TimeLLM_Stock_Directional.sh

accelerate launch --num_processes 1 --main_process_port 10100 run_timellm_directional.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/stock/ \
  --data_path vcb_stock_indicators_v2.csv \
  --model_id VCB_stock_60_1 \
  --model TimeLLM \
  --data Stock \
  --features MS \
  --target "Adj Close" \
  --freq d \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 1 \
  --enc_in 13 \
  --dec_in 13 \
  --c_out 1 \
  --d_model 32 \
  --d_ff 128 \
  --dropout 0.2 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 6 \
  --train_epochs 30 \
  --patience 7 \
  --patching_mode frequency_aware \
  --loss_type directional \
  --direction_weight 0.3 \
  --prompt_domain 1 \
  --model_comment TimeLLM-FFT-Directional
