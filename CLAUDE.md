# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A fork of Time-LLM (ICLR'24, reprograms a frozen LLM for time-series forecasting) adapted for **VCB (Vietcombank) stock price prediction**. The upstream code (`run_main.py`, `run_m4.py`, `run_pretrain.py`, `scripts/TimeLLM_ETT*.sh`, etc.) is kept; the stock work is layered on top as several experiment generations. There is no test suite or linter — validation is done by training and `evaluate_stock_model.py`.

Setup: Python 3.11, `pip install -r requirements.txt`. Training uses `accelerate launch` (or plain `python`) with GPT2 as the default backbone (`--llm_model GPT2 --llm_dim 768 --llm_layers 6`); `ds_config_zero2.json` is for DeepSpeed.

## New pipeline (thesis work, branch `thesis/multimodal`)

Active development happens in `stock_llm/` + `data_collection/`; the `run_stock_*.py` trainers below are the legacy generation. Use the project venv: `.venv/bin/python` (vLLM needs `PATH="$PWD/.venv/bin:$PATH"` and `VLLM_USE_FLASHINFER_SAMPLER=0`).

- Universe: 8 tickers in `stock_llm/data/universe.py` (ACB CTG FPT HPG MBB SSI VCB VNM).
- Collect (CafeF, ~1 req/s, writes gitignored `data/`): `data_collection/fetch_prices.py`, `fetch_news.py` (ticker pages), `fetch_market_news.py` (stock/bank/macro categories, resumable), `fetch_flows.py` (foreign + proprietary flows, VN-Index). Run from the repo root.
- Prices: `stock_llm/data/prices.py` builds the adjusted series from CafeF's `Change` (move vs adjusted reference price), not CafeF's `Adj Close`; days beyond any legal price band are unadjusted corporate actions and are zeroed.
- Text: `python -m stock_llm.text.corpus` -> `embed` (BGE-M3) -> `llm_score` (Qwen2.5-7B zero-shot sentiment per headline x tagged ticker). Embeddings/scores are cached by URL.
- News alignment (`stock_llm/data/news.py`): an item counts for day t only if published before 09:00 on t; date-only timestamps are treated as end of day.
- Train/eval: `python -m stock_llm.train --model {lstm,dlinear,patchtst} [--fold 0|1|2] [--min_move x] --seeds 0 1 2`. Folds are rolling-origin test years 2023 / 2024 / 2025-26; test is evaluated once with the val-selected checkpoint. Report winrate with trade count, next to the always-up/always-down floors; never filter test days by realised move size (that is look-ahead).
- `stock_llm/analysis/` holds the predictability studies (technical features show no signal; LLM news sentiment does).
- Event-driven pipeline: `python -m stock_llm.text.event_type` -> `stock_llm/data/events.py` (`build_events`, pickled to `data/text/events.pkl` + `events_cols.json`; targets `ret_H`, `exc_H` vs VN-Index, `rel_H` vs the 8-ticker peer average) -> `train_llm.py` (LoRA 7B, overfits) / `frozen_combiner.py` (frozen LLM features + logit/GBM).
- Thesis model Time-LLM+ (`models/timellm_plus.py`, `train_plus.py`): frozen backbone + C1 news prompt/knowledge tokens, C2 text-conditioned FFT patching, C3 decomposed selective loss; each axis is one flag (`--prompt/--knowledge/--patching/--loss`). Grid: `bash stock_llm/run_ablations.sh [c1|c2|c3|full|all]` (env BACKBONE/FOLDS/SEEDS); table: `python -m stock_llm.analysis.summarize_plus`. Smoke test with `--limit 16 --epochs 1`.
- Shell is zsh: unquoted `$var` is not word-split; use `${=var}` in loops, and avoid `pgrep -f`/`pkill -f` patterns that also match the calling shell's own command line.

## Data (legacy pipeline)

- `dataset/dataset/*` is gitignored. Raw prices are in `gia_VCB_2020-2025.csv` at the repo root.
- Generate datasets into `./dataset/dataset/stock/`: `python stock_data_preparation.py` (6 features -> `vcb_stock_indicators.csv`) or `stock_data_preparation_v2.py` (13 features incl. momentum -> `vcb_stock_indicators_v2.csv`; `_v0` is the oldest variant).
- Optional per-sample "professor advice" prompts: `python prompt_generator.py --data_path ... --output_dir ./dataset/dataset/stock [--use_api]` -> `prompts_short_term.json` / `prompts_mid_term.json` (API mode needs `OPENAI_API_KEY`; otherwise rule-based text).
- `STOCK_PREDICTION_GUIDE.txt` documents this pipeline end to end.

## Experiment generations (parallel, not layered)

Each generation has its own `run_stock_*.py` trainer, usually its own model and shell script. Pick the matching trio; mixing them breaks (e.g. `enc_in` must match the data file's feature count: 6 for v0/v1/FFT, 13 for v2/v3).

| Trainer | Model (`--model`) | Script | Notes |
|---|---|---|---|
| `run_stock_training_v0/v1/v2.py`, `run_stock_training.py` | `TimeLLM` / `TimeLLM_Stock` | `scripts/TimeLLM_Stock_V*.sh`, `_ShortTerm/_MidTerm.sh` | incremental: dynamic prompts, directional loss, momentum features |
| `run_stock_training_v3.py` | `TimeLLM_Stock_V3` | `scripts/TimeLLM_Stock_V3.sh [short_term\|mid_term] [dir_weight] [attn_type] [patch_preset]` | fixed multi-scale patches (20/40/50 d from FFT analysis), `FeatureAttention`, seq_len 120 |
| `run_stock_fft.py` | `TimeLLM` | `scripts/TimeLLM_Stock_FFT.sh [frequency_aware\|multi_scale\|single]` | FFT + attention patching only, run via `accelerate launch` |
| `run_stock_all_features.py` | `TimeLLM` | — | everything combined: dynamic prompts + FFT patching + `DirectionalLoss`; reports win rate on train/valid/test |

Run a script from the repo root, e.g. `bash scripts/TimeLLM_Stock_FFT.sh multi_scale`. Short-term = `pred_len 1`; mid-term = `pred_len 60`.

## Architecture

- `models/TimeLLM.py` is the **main, modified** model: besides the upstream reprogramming flow (patch embed -> `ReprogrammingLayer` cross-attention against LLM word-embedding prototypes -> frozen LLM -> `FlattenHead`), it adds `FrequencyAwarePatchBlock` (FFT-derived dominant periods + attention choose/learn patch lengths, with a residual connection added to fight vanishing gradients) and a `patching_mode` config switch (`frequency_aware` | `multi_scale` | `single`), plus a `dynamic_prompts` argument to `forward`/`forecast` for per-sample text prompts. `get_scale_analysis`, `get_learnable_patch_params`, `print_patch_info` expose the learned patch parameters (see `analyze_patch_lengths.py`).
- `models/TimeLLM_Stock.py` and `TimeLLM_Stock_V3.py` are standalone older model variants (V3 holds its own `FeatureAttention`, `ReprogrammingLayer`, `FlattenHead`). Models are loaded via `from models import ...` and the module's `Model` class.
- `data_provider/stock_data_loader.py` holds `Dataset_Stock` and `Dataset_Stock_WithPrompt` (attaches prompts per window) with `stock_collate_fn`; `data_factory.py` dispatches on `--data Stock`. Prompt-bank text for the static prompt is in `dataset/prompt_bank/Stock.txt`.
- Features mode `MS` with target `Adj Close`: multivariate input, single-series output (`c_out 1`).
- `evaluate_stock_model.py --checkpoint_path ... --version v0|v1|v2|v3 [--use_dynamic_prompt] --prediction_type short_term` computes MSE/RMSE/MAE/MAPE/R², directional accuracy, and a simulated trading P&L/Sharpe/drawdown, and writes prediction/trading plots (examples in `report/`). Its model args must match training.
- Checkpoints go to `./checkpoints/`.
