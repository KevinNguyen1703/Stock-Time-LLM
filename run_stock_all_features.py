"""
Stock Prediction Training - All Features Combined
Combines:
1. Dynamic Prompts (expert analysis per sample)
2. FFT + Attention (frequency-aware patching)
3. Directional Loss

Outputs: Winrate for train, valid, test datasets
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
    Combined loss: MSE + Soft Directional penalty (BCE-like)
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
            # Soft directional loss (differentiable)
            scale = 10.0
            pred_dir_prob = torch.sigmoid(scale * (pred_first - prev_values))
            target_dir_prob = torch.sigmoid(scale * (target_first - prev_values))
            
            eps = 1e-7
            direction_loss = -torch.mean(
                target_dir_prob * torch.log(pred_dir_prob + eps) +
                (1 - target_dir_prob) * torch.log(1 - pred_dir_prob + eps)
            )
        else:
            # Hard directional loss
            pred_direction = torch.sign(pred_first - prev_values)
            target_direction = torch.sign(target_first - prev_values)
            direction_mismatch = (pred_direction != target_direction).float()
            direction_loss = direction_mismatch.mean()
        
        return mse_loss + self.direction_weight * direction_loss
    
    def compute_direction_accuracy(self, pred, target, prev_values):
        """Compute directional accuracy (winrate)"""
        pred_first = pred[:, 0, 0]
        target_first = target[:, 0, 0]
        
        pred_up = (pred_first > prev_values)
        actual_up = (target_first > prev_values)
        
        correct = (pred_up == actual_up).sum().item()
        total = pred_up.numel()
        
        return correct, total, correct / total * 100 if total > 0 else 0


def evaluate_with_winrate(args, accelerator, model, data_loader, criterion, desc="Eval"):
    """Evaluate and compute winrate"""
    total_loss = []
    total_correct = 0
    total_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, disable=not accelerator.is_local_main_process):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                prompts = None
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # Forward with dynamic prompts
            if prompts is not None:
                unwrapped = accelerator.unwrap_model(model)
                outputs = unwrapped(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            prev_values = batch_x[:, -1, f_dim].to(accelerator.device)
            
            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion.mse(pred, true)
            total_loss.append(loss.item())
            
            # Compute winrate
            correct, total, _ = criterion.compute_direction_accuracy(pred, true, prev_values)
            total_correct += correct
            total_samples += total

    avg_loss = np.average(total_loss)
    winrate = total_correct / total_samples * 100 if total_samples > 0 else 0

    model.train()
    return avg_loss, winrate


def parse_args():
    parser = argparse.ArgumentParser(description='Time-LLM Stock - All Features')
    
    # Feature toggles
    parser.add_argument('--use_dynamic_prompt', action='store_true', default=True,
                        help='Use dynamic prompts from data loader')
    parser.add_argument('--patching_mode', type=str, default='frequency_aware',
                        choices=['frequency_aware', 'multi_scale', 'single'],
                        help='Patching mode: frequency_aware (FFT+attention), multi_scale, single')
    parser.add_argument('--direction_weight', type=float, default=0.3,
                        help='Weight for directional loss (0 = MSE only)')
    parser.add_argument('--use_soft_direction', action='store_true', default=True,
                        help='Use soft (differentiable) directional loss')
    
    # Prompt config
    parser.add_argument('--prompt_data_path', type=str, default=None,
                        help='Path to prompt JSON file')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='VCB_stock_all')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-AllFeatures')
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
    
    # Model config
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
    
    # Build model ID
    args.model_id = f'{args.model_id}_{args.seq_len}_{args.pred_len}'
    
    return args


def train_all_features(args):
    """Training with all features: dynamic prompt + FFT + directional loss"""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_pm{}_dw{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.d_ff, args.patching_mode, args.direction_weight)
    
    # Load content for prompts
    args.content = load_content(args)
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Time-LLM Stock Prediction - ALL FEATURES")
    accelerator.print(f"{'='*70}")
    accelerator.print(f"Dynamic Prompts: {args.use_dynamic_prompt}")
    accelerator.print(f"Patching Mode: {args.patching_mode} (FFT+Attention)")
    accelerator.print(f"Directional Loss Weight: {args.direction_weight}")
    accelerator.print(f"Data: {args.data_path}")
    accelerator.print(f"{'='*70}\n")
    
    # Load data with prompts
    train_data, train_loader = data_provider(args, 'train', with_prompt=args.use_dynamic_prompt)
    vali_data, vali_loader = data_provider(args, 'val', with_prompt=args.use_dynamic_prompt)
    test_data, test_loader = data_provider(args, 'test', with_prompt=args.use_dynamic_prompt)
    
    # Initialize model with patching_mode
    model = TimeLLM.Model(args).float()
    accelerator.print(f"Model: {args.model} | Patching: {args.patching_mode} | Features: {args.enc_in}")
    
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
    
    # Directional loss
    criterion = DirectionalLoss(
        direction_weight=args.direction_weight,
        use_soft_direction=args.use_soft_direction
    )
    
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    training_history = {
        'train_loss': [], 'vali_loss': [], 'test_loss': [],
        'train_winrate': [], 'vali_winrate': [], 'test_winrate': []
    }
    
    best_winrate = 0
    
    for epoch in range(args.train_epochs):
        train_loss = []
        train_correct = 0
        train_total = 0
        
        model.train()
        epoch_time = time.time()
        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), 
                             desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                prompts = None
            
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # Forward with dynamic prompts
            if prompts is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                outputs = unwrapped_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
            
            prev_values = batch_x[:, -1, f_dim]
            
            # Compute loss with directional component
            loss = criterion(outputs, batch_y_target, prev_values)
            train_loss.append(loss.item())
            
            # Track winrate
            correct, total, _ = criterion.compute_direction_accuracy(
                outputs.detach(), batch_y_target.detach(), prev_values)
            train_correct += correct
            train_total += total
            
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(trained_parameters, max_norm=1.0)
            model_optim.step()
            scheduler.step()
        
        train_loss_avg = np.average(train_loss)
        train_winrate = train_correct / train_total * 100 if train_total > 0 else 0
        
        # Evaluate
        vali_loss, vali_winrate = evaluate_with_winrate(args, accelerator, model, vali_loader, criterion, "Valid")
        test_loss, test_winrate = evaluate_with_winrate(args, accelerator, model, test_loader, criterion, "Test")
        
        accelerator.print(f"\nEpoch {epoch+1} | Time: {time.time()-epoch_time:.2f}s")
        accelerator.print(f"  Train Loss: {train_loss_avg:.6f} | Train Winrate: {train_winrate:.2f}%")
        accelerator.print(f"  Valid Loss: {vali_loss:.6f} | Valid Winrate: {vali_winrate:.2f}%")
        accelerator.print(f"  Test  Loss: {test_loss:.6f} | Test  Winrate: {test_winrate:.2f}%")
        
        training_history['train_loss'].append(float(train_loss_avg))
        training_history['vali_loss'].append(float(vali_loss))
        training_history['test_loss'].append(float(test_loss))
        training_history['train_winrate'].append(float(train_winrate))
        training_history['vali_winrate'].append(float(vali_winrate))
        training_history['test_winrate'].append(float(test_winrate))
        
        if vali_winrate > best_winrate:
            best_winrate = vali_winrate
            accelerator.print(f"  >> New best validation winrate: {best_winrate:.2f}%")
        
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break
        
        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
    
    # Save history
    if accelerator.is_local_main_process:
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        config_dict = {k: v for k, v in vars(args).items() if k != 'content'}
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    accelerator.wait_for_everyone()
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Training Completed!")
    accelerator.print(f"Best Validation Winrate: {best_winrate:.2f}%")
    accelerator.print(f"Model saved to: {path}")
    accelerator.print(f"{'='*70}")
    
    return path, training_history


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path, history = train_all_features(args)

