"""
Stock Price Prediction Training V0 - Baseline
Uses original Time-LLM without any enhancements:
- Standard MSE loss only (no directional loss)
- Static prompts (no dynamic per-sample prompts)
- Basic indicators only (5 features)

For comparison with enhanced V2 version
"""

import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM  # Use original TimeLLM, not TimeLLM_Stock
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, load_content


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    """Standard validation without directional metrics"""
    total_loss = []
    total_mae_loss = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(vali_loader, desc="Validating"):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Time-LLM Stock Prediction V0 - Baseline')
    
    parser.add_argument('--prediction_type', type=str, default='short_term',
                        choices=['short_term', 'mid_term'])
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='VCB_stock_v0_baseline')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-V0-Baseline')
    parser.add_argument('--model', type=str, default='TimeLLM')  # Original TimeLLM
    parser.add_argument('--seed', type=int, default=2021)
    
    # Data config - V0 uses basic indicators
    parser.add_argument('--data', type=str, default='Stock')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/stock/')
    parser.add_argument('--data_path', type=str, default='vcb_stock_indicators_v0.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='Adj Close')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Forecasting config
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--label_len', type=int, default=30)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    
    # Model config - V0 has 5 input features (basic indicators)
    parser.add_argument('--enc_in', type=int, default=5)
    parser.add_argument('--dec_in', type=int, default=5)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)  # Standard dropout
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--prompt_domain', type=int, default=1)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)
    
    # Training config - Standard settings
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--loader', type=str, default='modal')
    
    args = parser.parse_args()
    
    # Set prediction length based on type
    if args.prediction_type == 'short_term':
        args.pred_len = 1
        args.model_id = f'{args.model_id}_60_1'
    else:
        args.pred_len = 60
        args.model_id = f'{args.model_id}_60_60'
        args.batch_size = 8
        args.learning_rate = 0.0005
    
    return args


def train_model_v0(args):
    """Training with standard MSE loss (no enhancements)"""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.d_ff)
    
    # Load content (static prompt)
    args.content = load_content(args)
    
    # Load data WITHOUT dynamic prompts
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Time-LLM Stock Prediction V0 - BASELINE")
    accelerator.print(f"{'='*60}")
    accelerator.print(f"Data: {args.data_path}")
    accelerator.print(f"Model: {args.model} (Original, no enhancements)")
    accelerator.print(f"Loss: Standard MSE (no directional component)")
    accelerator.print(f"Prompts: Static (no per-sample dynamic prompts)")
    accelerator.print(f"{'='*60}\n")
    
    # Load data without prompts
    train_data, train_loader = data_provider(args, 'train', with_prompt=False)
    vali_data, vali_loader = data_provider(args, 'val', with_prompt=False)
    test_data, test_loader = data_provider(args, 'test', with_prompt=False)
    
    # Initialize original TimeLLM model
    model = TimeLLM.Model(args).float()
    accelerator.print(f"Model initialized with {args.enc_in} input features")
    
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)
    
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate
    )
    
    # Standard MSE loss only (no directional component)
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    training_history = {
        'train_loss': [], 'vali_loss': [], 'test_loss': [],
        'vali_mae': [], 'test_mae': []
    }
    
    accelerator.print(f"Starting V0 Baseline Training...")
    accelerator.print(f"Train samples: {len(train_data)}, Val: {len(vali_data)}, Test: {len(test_data)}\n")
    
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        
        model.train()
        epoch_time = time.time()
        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
            
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # Standard forward pass
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
            
            # Standard MSE loss
            loss = criterion(outputs, batch_y_target)
            train_loss.append(loss.item())
            
            accelerator.backward(loss)
            model_optim.step()
            scheduler.step()
        
        train_loss_avg = np.average(train_loss)
        
        accelerator.print(f"Epoch {epoch+1} | Time: {time.time()-epoch_time:.2f}s | Train Loss: {train_loss_avg:.6f}")
        
        # Validation
        vali_loss, vali_mae = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        
        accelerator.print(
            f"         | Vali Loss: {vali_loss:.6f} | Vali MAE: {vali_mae:.6f} | "
            f"Test Loss: {test_loss:.6f} | Test MAE: {test_mae:.6f}"
        )
        
        training_history['train_loss'].append(float(train_loss_avg))
        training_history['vali_loss'].append(float(vali_loss))
        training_history['test_loss'].append(float(test_loss))
        training_history['vali_mae'].append(float(vali_mae))
        training_history['test_mae'].append(float(test_mae))
        
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break
        
        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
    
    # Save training history
    if accelerator.is_local_main_process:
        with open(os.path.join(path, 'training_history_v0.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save config
        config_dict = {k: v for k, v in vars(args).items() if k != 'content'}
        config_dict['version'] = 'v0_baseline'
        config_dict['enhancements'] = 'None (baseline)'
        with open(os.path.join(path, 'config_v0.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    accelerator.wait_for_everyone()
    
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"V0 Baseline Training completed!")
    accelerator.print(f"Model saved to: {path}")
    accelerator.print(f"{'='*60}")
    
    return path, training_history


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path, history = train_model_v0(args)

