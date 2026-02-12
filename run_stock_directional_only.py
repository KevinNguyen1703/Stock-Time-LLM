"""
Stock Prediction - Directional Loss ONLY
Evaluates the impact of directional loss in isolation.

Uses:
- Original TimeLLM model (single patching mode, no FFT+attention)
- NO dynamic prompts
- Directional Loss (configurable weight)

Outputs: MSE, MAE, Winrate for train, valid, test
"""

import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, load_content


class DirectionalLoss(nn.Module):
    """
    Combined loss: MSE + Soft Directional penalty
    Set direction_weight=0 for pure MSE (baseline)
    """
    def __init__(self, direction_weight=0.3, use_soft_direction=True):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
        self.use_soft_direction = use_soft_direction
    
    def forward(self, pred, target, prev_values=None):
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        pred_first = pred[:, 0, 0]
        target_first = target[:, 0, 0]
        
        if self.use_soft_direction:
            # Soft directional loss (differentiable BCE-like)
            scale = 10.0
            pred_dir_prob = torch.sigmoid(scale * (pred_first - prev_values))
            target_dir_prob = torch.sigmoid(scale * (target_first - prev_values))
            
            eps = 1e-7
            direction_loss = -torch.mean(
                target_dir_prob * torch.log(pred_dir_prob + eps) +
                (1 - target_dir_prob) * torch.log(1 - pred_dir_prob + eps)
            )
        else:
            # Hard directional loss (non-differentiable)
            pred_direction = torch.sign(pred_first - prev_values)
            target_direction = torch.sign(target_first - prev_values)
            direction_mismatch = (pred_direction != target_direction).float()
            direction_loss = direction_mismatch.mean()
        
        return mse_loss + self.direction_weight * direction_loss


def compute_metrics(pred, target, prev_values):
    """Compute MSE, MAE, and Winrate"""
    mse = nn.MSELoss()(pred, target).item()
    mae = nn.L1Loss()(pred, target).item()
    
    pred_first = pred[:, 0, 0]
    target_first = target[:, 0, 0]
    pred_up = (pred_first > prev_values)
    actual_up = (target_first > prev_values)
    correct = (pred_up == actual_up).sum().item()
    total = pred_up.numel()
    winrate = correct / total * 100 if total > 0 else 0
    
    return mse, mae, winrate, correct, total


def evaluate(args, accelerator, model, data_loader, desc="Eval"):
    """Evaluate model"""
    all_mse, all_mae = [], []
    total_correct, total_samples = 0, 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, disable=not accelerator.is_local_main_process):
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
            prev_values = batch_x[:, -1, f_dim].to(accelerator.device)
            
            mse, mae, _, correct, total = compute_metrics(
                outputs.detach(), batch_y.detach(), prev_values)
            all_mse.append(mse)
            all_mae.append(mae)
            total_correct += correct
            total_samples += total

    avg_mse = np.average(all_mse)
    avg_mae = np.average(all_mae)
    winrate = total_correct / total_samples * 100 if total_samples > 0 else 0

    model.train()
    return avg_mse, avg_mae, winrate


def parse_args():
    parser = argparse.ArgumentParser(description='Time-LLM Stock - Directional Loss Only')
    
    # Directional loss config
    parser.add_argument('--direction_weight', type=float, default=0.3,
                        help='Weight for directional loss (0 = pure MSE baseline)')
    parser.add_argument('--use_soft_direction', action='store_true', default=True,
                        help='Use soft (differentiable) directional loss')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='VCB_stock_dir')
    parser.add_argument('--model_comment', type=str, default='DirectionalOnly')
    parser.add_argument('--model', type=str, default='TimeLLM')
    parser.add_argument('--seed', type=int, default=2021)
    
    # Data config
    parser.add_argument('--data', type=str, default='Stock')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/stock/')
    parser.add_argument('--data_path', type=str, default='vcb_stock_indicators.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='Adj Close')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Forecasting config
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--label_len', type=int, default=30)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    
    # Model config - use SINGLE patching (original, no FFT)
    parser.add_argument('--patching_mode', type=str, default='single',
                        help='Use single patching (original) to isolate directional loss effect')
    parser.add_argument('--enc_in', type=int, default=6)
    parser.add_argument('--dec_in', type=int, default=6)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--prompt_domain', type=int, default=1)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)
    
    # Training config
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=30)
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
    args.model_id = f'{args.model_id}_{args.seq_len}_{args.pred_len}'
    
    return args


def train(args):
    """Training with directional loss only"""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    loss_type = "MSE" if args.direction_weight == 0 else f"Directional(w={args.direction_weight})"
    
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_dw{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.direction_weight)
    
    args.content = load_content(args)
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Time-LLM Stock - DIRECTIONAL LOSS EVALUATION")
    accelerator.print(f"{'='*70}")
    accelerator.print(f"Loss: {loss_type}")
    accelerator.print(f"Patching: {args.patching_mode} (original, no FFT)")
    accelerator.print(f"Dynamic Prompts: DISABLED")
    accelerator.print(f"Data: {args.data_path}")
    accelerator.print(f"{'='*70}\n")
    
    # Load data WITHOUT prompts
    train_data, train_loader = data_provider(args, 'train', with_prompt=False)
    vali_data, vali_loader = data_provider(args, 'val', with_prompt=False)
    test_data, test_loader = data_provider(args, 'test', with_prompt=False)
    
    # Model with single patching mode (original)
    model = TimeLLM.Model(args).float()
    accelerator.print(f"Model: {args.model} | Patching: {args.patching_mode}")
    
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
    
    # Directional loss (or pure MSE if direction_weight=0)
    criterion = DirectionalLoss(
        direction_weight=args.direction_weight,
        use_soft_direction=args.use_soft_direction
    )
    
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    training_history = {
        'train_mse': [], 'train_mae': [], 'train_winrate': [],
        'vali_mse': [], 'vali_mae': [], 'vali_winrate': [],
        'test_mse': [], 'test_mae': [], 'test_winrate': []
    }
    
    best_winrate = 0
    
    for epoch in range(args.train_epochs):
        train_loss = []
        train_correct, train_total = 0, 0
        
        model.train()
        epoch_time = time.time()
        
        for batch in tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process):
            i, batch_data = batch[0], batch[1]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data[:4]
            
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
            prev_values = batch_x[:, -1, f_dim]
            
            loss = criterion(outputs, batch_y_target, prev_values)
            train_loss.append(loss.item())
            
            # Track winrate
            pred_up = (outputs[:, 0, 0] > prev_values)
            actual_up = (batch_y_target[:, 0, 0] > prev_values)
            train_correct += (pred_up == actual_up).sum().item()
            train_total += outputs.shape[0]
            
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(trained_parameters, max_norm=1.0)
            model_optim.step()
            scheduler.step()
        
        train_loss_avg = np.average(train_loss)
        train_winrate = train_correct / train_total * 100 if train_total > 0 else 0
        
        # Evaluate
        vali_mse, vali_mae, vali_winrate = evaluate(args, accelerator, model, vali_loader, "Valid")
        test_mse, test_mae, test_winrate = evaluate(args, accelerator, model, test_loader, "Test")
        
        accelerator.print(f"\nEpoch {epoch+1} | Time: {time.time()-epoch_time:.2f}s")
        accelerator.print(f"  Train Loss: {train_loss_avg:.6f} | Train Winrate: {train_winrate:.2f}%")
        accelerator.print(f"  Valid MSE: {vali_mse:.6f} | Valid MAE: {vali_mae:.6f} | Valid Winrate: {vali_winrate:.2f}%")
        accelerator.print(f"  Test  MSE: {test_mse:.6f} | Test  MAE: {test_mae:.6f} | Test  Winrate: {test_winrate:.2f}%")
        
        training_history['train_mse'].append(float(train_loss_avg))
        training_history['train_winrate'].append(float(train_winrate))
        training_history['vali_mse'].append(float(vali_mse))
        training_history['vali_mae'].append(float(vali_mae))
        training_history['vali_winrate'].append(float(vali_winrate))
        training_history['test_mse'].append(float(test_mse))
        training_history['test_mae'].append(float(test_mae))
        training_history['test_winrate'].append(float(test_winrate))
        
        if vali_winrate > best_winrate:
            best_winrate = vali_winrate
            accelerator.print(f"  >> New best validation winrate: {best_winrate:.2f}%")
        
        early_stopping(vali_mse, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break
        
        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
    
    # Save
    if accelerator.is_local_main_process:
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        config_dict = {k: v for k, v in vars(args).items() if k != 'content'}
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    accelerator.wait_for_everyone()
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Training Completed!")
    accelerator.print(f"Loss Type: {loss_type}")
    accelerator.print(f"Best Validation Winrate: {best_winrate:.2f}%")
    accelerator.print(f"Model saved to: {path}")
    accelerator.print(f"{'='*70}")
    
    return path, training_history


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path, history = train(args)

