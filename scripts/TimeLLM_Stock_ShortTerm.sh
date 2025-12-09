#!/bin/bash

# Time-LLM Training Script for VCB Stock - Short Term Prediction
# Predicts next 1 day based on last 60 days of technical indicators

model_name=TimeLLM
train_epochs=50
learning_rate=0.001
llama_layers=6

master_port=2025
num_process=1
batch_size=16
d_model=32
d_ff=128

comment='TimeLLM-Stock-ShortTerm'

# Short-term prediction: seq_len=60, pred_len=1
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/stock/ \
  --data_path vcb_stock_indicators.csv \
  --model_id VCB_stock_60_1 \
  --model $model_name \
  --data Stock \
  --features MS \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --target 'Adj Close' \
  --freq 'd' \
  --patch_len 8 \
  --stride 4 \
  --prompt_domain 1 \
  --llm_model GPT2 \
  --llm_dim 768

