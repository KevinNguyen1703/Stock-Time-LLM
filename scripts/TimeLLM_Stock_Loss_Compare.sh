#!/bin/bash
# ============================================
# LOSS FUNCTION COMPARISON
# Run all 4 loss types and compare winrates
# ============================================

# Common parameters
COMMON_PARAMS="--patching_mode single \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/stock/ \
  --data_path vcb_stock_indicators.csv \
  --model TimeLLM \
  --data Stock \
  --features MS \
  --target \"Adj Close\" \
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
  --prompt_domain 1"

echo "============================================"
echo "LOSS FUNCTION COMPARISON"
echo "============================================"
echo ""
echo "Will run 4 experiments:"
echo "1. MSE (baseline)"
echo "2. Directional Loss"
echo "3. Asymmetric Loss"
echo "4. Weighted Directional Loss"
echo ""
echo "============================================"

# 1. MSE Baseline
echo ""
echo "[1/4] Running MSE (baseline)..."
echo "============================================"
accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2030 run_stock_directional_only.py \
  --loss_type mse \
  --model_id VCB_loss_mse \
  --model_comment "Loss-MSE" \
  $COMMON_PARAMS

# 2. Directional Loss
echo ""
echo "[2/4] Running Directional Loss..."
echo "============================================"
accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2031 run_stock_directional_only.py \
  --loss_type directional \
  --direction_weight 0.3 \
  --model_id VCB_loss_directional \
  --model_comment "Loss-Directional" \
  $COMMON_PARAMS

# 3. Asymmetric Loss
echo ""
echo "[3/4] Running Asymmetric Loss..."
echo "============================================"
accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2032 run_stock_directional_only.py \
  --loss_type asymmetric \
  --up_penalty 1.5 \
  --down_penalty 1.0 \
  --model_id VCB_loss_asymmetric \
  --model_comment "Loss-Asymmetric" \
  $COMMON_PARAMS

# 4. Weighted Directional Loss
echo ""
echo "[4/4] Running Weighted Directional Loss..."
echo "============================================"
accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2033 run_stock_directional_only.py \
  --loss_type weighted_dir \
  --direction_weight 0.5 \
  --model_id VCB_loss_weighted \
  --model_comment "Loss-WeightedDir" \
  $COMMON_PARAMS

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "Check ./checkpoints/ for results"
echo "============================================"

