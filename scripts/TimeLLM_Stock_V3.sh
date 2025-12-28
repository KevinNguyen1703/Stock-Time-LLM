#!/bin/bash

# Time-LLM Stock Prediction V3 - FFT-OPTIMIZED CONFIGURATION
# Combines ALL improvements:
# - FFT-based multi-scale patching (20, 40, 50 day windows) - captures dominant cycles
# - Feature attention for interpretability
# - Enhanced trend prompts with momentum analysis
# - V2 data with momentum features
# - ChatGPT-generated dynamic prompts
# - Directional loss
# - Extended seq_len=120 to capture 40-50 day patterns

# Configuration
model_name=TimeLLM_Stock_V3
prediction_type=${1:-short_term}
direction_weight=${2:-0.3}
attention_type=${3:-additive}
patch_preset=${4:-fft_vcb}

echo "========================================================================"
echo "Time-LLM Stock Prediction V3 - FFT-OPTIMIZED"
echo "========================================================================"
echo "Prediction Type: $prediction_type"
echo "Direction Weight: $direction_weight"
echo "Attention Type: $attention_type"
echo "Patch Preset: $patch_preset"
echo ""
echo "V3 Enhancements (FFT-Optimized):"
echo "  ✓ FFT-based multi-scale patching (20, 40, 50 day windows)"
echo "  ✓ Extended seq_len=120 (captures bi-monthly patterns)"
echo "  ✓ Feature attention ($attention_type)"
echo "  ✓ Enhanced trend prompts with momentum"
echo "  ✓ V2 data (13 features including momentum)"
echo "  ✓ ChatGPT-generated dynamic prompts"
echo "  ✓ Directional loss (weight=$direction_weight)"
echo "========================================================================"

# Settings - FFT-OPTIMIZED
llm_model=GPT2
llm_dim=768
llm_layers=6
d_model=32
d_ff=128
batch_size=16
learning_rate=0.0005
dropout=0.2
train_epochs=30
patience=7

# FFT-optimized: seq_len=120 to capture 40-50 day dominant patterns
seq_len=120
label_len=60

# Set pred_len based on prediction type
if [ "$prediction_type" == "short_term" ]; then
    pred_len=1
    data_path="vcb_stock_indicators_v2.csv"
    prompt_data_path="prompts_short_term.json"  # ChatGPT prompts
    model_id="VCB_v3_fft_${seq_len}_1"
else
    pred_len=60
    data_path="vcb_stock_indicators_v2.csv"
    prompt_data_path="prompts_mid_term.json"  # ChatGPT prompts
    model_id="VCB_v3_fft_${seq_len}_60"
    batch_size=8
    learning_rate=0.0003
fi

# V3 uses 13 input features (V2 data)
enc_in=13
dec_in=13
c_out=1

echo ""
echo "Model Configuration:"
echo "  - Input features: $enc_in"
echo "  - Sequence length: $seq_len (FFT-optimized)"
echo "  - Label length: $label_len"
echo "  - Prediction length: $pred_len"
echo "  - Patch preset: $patch_preset"
echo "  - Data file: $data_path"
echo "  - Prompt file: $prompt_data_path (ChatGPT)"
echo ""
echo "FFT Analysis Results for VCB Stock:"
echo "  - Top dominant periods: 40d, 50d, 29-30d"
echo "  - Recommended patches: [20, 40, 50] days"
echo "  - Captures: monthly, bi-monthly, quarterly patterns"
echo ""

# Run training
python run_stock_training_v3.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id $model_id \
    --model $model_name \
    --data Stock \
    --root_path ./dataset/dataset/stock/ \
    --data_path $data_path \
    --prompt_data_path $prompt_data_path \
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
    --direction_weight $direction_weight \
    --patch_preset $patch_preset \
    --use_dynamic_prompt \
    --use_multi_scale \
    --use_feature_attention \
    --attention_type $attention_type \
    --model_comment "V3-FFT-Optimized"

echo ""
echo "========================================================================"
echo "V3 FFT-Optimized Training Complete!"
echo "========================================================================"
