"""
Stock Prediction - Loss Function Comparison
Evaluates different loss functions using winrate.

Supported Loss Types:
1. mse          - Pure MSE (baseline)
2. directional  - MSE + Soft Directional penalty (BCE-like)
3. asymmetric   - MSE with asymmetric penalties for UP/DOWN errors
4. weighted_dir - MSE + Direction penalty weighted by move magnitude

Uses:
- Original TimeLLM model (single patching mode, no FFT+attention)
- NO dynamic prompts

Outputs: MSE, MAE, Winrate for train, valid, test
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MSELossWrapper(nn.Module):
    """Pure MSE Loss (baseline)"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, prev_values=None):
        return self.mse(pred, target)


class DirectionalLoss(nn.Module):
    """
    MSE + Soft Directional penalty (BCE-like)
    Penalizes wrong direction predictions in a differentiable way.
    """
    def __init__(self, direction_weight=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
    
    def forward(self, pred, target, prev_values=None):
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        pred_first = pred[:, 0, 0]
        target_first = target[:, 0, 0]
        
        # Soft directional loss (differentiable BCE-like)
        scale = 10.0
        pred_dir_prob = torch.sigmoid(scale * (pred_first - prev_values))
        target_dir_prob = torch.sigmoid(scale * (target_first - prev_values))
        
        eps = 1e-7
        direction_loss = -torch.mean(
            target_dir_prob * torch.log(pred_dir_prob + eps) +
            (1 - target_dir_prob) * torch.log(1 - pred_dir_prob + eps)
        )
        
        return mse_loss + self.direction_weight * direction_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss - penalizes missed UP moves more than missed DOWN moves.
    Useful when missing a rally is more costly than missing a drop.
    """
    def __init__(self, up_penalty=1.5, down_penalty=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.up_penalty = up_penalty      # Penalty for missing UP (pred DOWN, actual UP)
        self.down_penalty = down_penalty  # Penalty for missing DOWN (pred UP, actual DOWN)
    
    def forward(self, pred, target, prev_values=None):
        # Base MSE per sample
        mse_per_sample = self.mse(pred, target).mean(dim=[1, 2])  # (batch,)
        
        if prev_values is None:
            return mse_per_sample.mean()
        
        pred_first = pred[:, 0, 0]
        target_first = target[:, 0, 0]
        
        pred_up = pred_first > prev_values
        actual_up = target_first > prev_values
        
        # Apply asymmetric weights
        weights = torch.ones_like(mse_per_sample)
        
        # Missed UP: predicted DOWN but actual UP
        missed_up = (~pred_up) & actual_up
        weights[missed_up] = self.up_penalty
        
        # Missed DOWN: predicted UP but actual DOWN
        missed_down = pred_up & (~actual_up)
        weights[missed_down] = self.down_penalty
        
        return (mse_per_sample * weights).mean()


class WeightedDirectionalLoss(nn.Module):
    """
    Weighted Directional Loss - direction penalty weighted by move magnitude.
    Penalizes wrong direction more when the actual move is large.
    """
    def __init__(self, direction_weight=0.5, magnitude_scale=10.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
        self.magnitude_scale = magnitude_scale
    
    def forward(self, pred, target, prev_values=None):
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        pred_first = pred[:, 0, 0]
        target_first = target[:, 0, 0]
        
        eps = 1e-8
        # Calculate returns
        pred_return = (pred_first - prev_values) / (torch.abs(prev_values) + eps)
        target_return = (target_first - prev_values) / (torch.abs(prev_values) + eps)
        
        # Direction match: +1 if same direction, -1 if opposite
        direction_match = torch.sign(pred_return) * torch.sign(target_return)
        
        # Weight by magnitude of actual move (larger moves = more important)
        magnitude_weight = torch.abs(target_return) * self.magnitude_scale
        
        # Penalty for wrong direction, weighted by magnitude
        direction_loss = torch.mean(F.relu(-direction_match) * (1 + magnitude_weight))
        
        return mse_loss + self.direction_weight * direction_loss


def get_loss_function(loss_type, direction_weight=0.3, up_penalty=1.5, down_penalty=1.0):
    """Factory function to get loss by name"""
    if loss_type == 'mse':
        return MSELossWrapper()
    elif loss_type == 'directional':
        return DirectionalLoss(direction_weight=direction_weight)
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(up_penalty=up_penalty, down_penalty=down_penalty)
    elif loss_type == 'weighted_dir':
        return WeightedDirectionalLoss(direction_weight=direction_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
    parser = argparse.ArgumentParser(description='Time-LLM Stock - Loss Function Comparison')
    
    # Loss function config
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'directional', 'asymmetric', 'weighted_dir'],
                        help='Loss function type: mse, directional, asymmetric, weighted_dir')
    parser.add_argument('--direction_weight', type=float, default=0.3,
                        help='Weight for directional component (for directional/weighted_dir)')
    parser.add_argument('--up_penalty', type=float, default=1.5,
                        help='Penalty for missing UP moves (for asymmetric loss)')
    parser.add_argument('--down_penalty', type=float, default=1.0,
                        help='Penalty for missing DOWN moves (for asymmetric loss)')
    
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
    """Training with configurable loss function"""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # Loss type description
    loss_desc = {
        'mse': 'MSE (baseline)',
        'directional': f'Directional (w={args.direction_weight})',
        'asymmetric': f'Asymmetric (up={args.up_penalty}, down={args.down_penalty})',
        'weighted_dir': f'Weighted Directional (w={args.direction_weight})'
    }
    
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_loss{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.loss_type)
    
    args.content = load_content(args)
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Time-LLM Stock - LOSS FUNCTION COMPARISON")
    accelerator.print(f"{'='*70}")
    accelerator.print(f"Loss Type: {args.loss_type} -> {loss_desc[args.loss_type]}")
    accelerator.print(f"Patching: {args.patching_mode}")
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
    
    # Get loss function by type
    criterion = get_loss_function(
        args.loss_type,
        direction_weight=args.direction_weight,
        up_penalty=args.up_penalty,
        down_penalty=args.down_penalty
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
    accelerator.print(f"Loss Type: {args.loss_type} -> {loss_desc[args.loss_type]}")
    accelerator.print(f"Best Validation Winrate: {best_winrate:.2f}%")
    accelerator.print(f"Model saved to: {path}")
    accelerator.print(f"{'='*70}")
    
    return path, training_history


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path, history = train(args)

