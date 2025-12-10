#!/bin/bash

# Time-LLM Stock Prediction V0 - Baseline Training Script
# Uses original Time-LLM without any enhancements for comparison

# Configuration
model_name=TimeLLM  # Original TimeLLM model
prediction_type=${1:-short_term}

echo "============================================"
echo "Time-LLM Stock Prediction V0 - BASELINE"
echo "============================================"
echo "Prediction Type: $prediction_type"
echo "Model: Original TimeLLM (no enhancements)"
echo "Loss: Standard MSE"
echo "Prompts: Static"
echo "============================================"

# Settings
llm_model=GPT2
llm_dim=768
llm_layers=6
d_model=32
d_ff=128
batch_size=16
learning_rate=0.001
dropout=0.1
train_epochs=50
patience=10
seq_len=60
label_len=30

# Set pred_len based on prediction type
if [ "$prediction_type" == "short_term" ]; then
    pred_len=1
    data_path="vcb_stock_indicators_v0.csv"
    model_id="VCB_v0_baseline_60_1"
else
    pred_len=60
    data_path="vcb_stock_indicators_v0.csv"
    model_id="VCB_v0_baseline_60_60"
    batch_size=8
    learning_rate=0.0005
fi

# V0 has 5 input features (basic indicators only)
enc_in=5
dec_in=5
c_out=1

echo ""
echo "Model Configuration:"
echo "  - Input features: $enc_in (RSI, MACD, BB_Position, Volume_Norm, ROC)"
echo "  - Sequence length: $seq_len"
echo "  - Prediction length: $pred_len"
echo "  - Data file: $data_path"
echo ""

# Run training
python run_stock_training_v0.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id $model_id \
    --model $model_name \
    --data Stock \
    --root_path ./dataset/dataset/stock/ \
    --data_path $data_path \
    --features MS \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --d_model $d_model \
    --d_ff $d_ff \
    --llm_model $llm_model \
    --llm_dim $llm_dim \
    --llm_layers $llm_layers \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --dropout $dropout \
    --train_epochs $train_epochs \
    --patience $patience \
    --prediction_type $prediction_type \
    --model_comment "V0-Baseline"

echo ""
echo "============================================"
echo "V0 Baseline Training Complete!"
echo "============================================"

