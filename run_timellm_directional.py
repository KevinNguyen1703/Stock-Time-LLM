"""
TimeLLM Training Script with Directional Loss Support
=======================================================
This script trains the TimeLLM model with FFT+Attention (frequency_aware patching)
and provides options for:
- With/Without directional loss for comparison
- Winrate and P&L evaluation on validation and test sets
- Comprehensive logging for comparison experiments

Usage:
    # Without directional loss (baseline)
    python run_timellm_directional.py --use_directional_loss 0
    
    # With directional loss
    python run_timellm_directional.py --use_directional_loss 1 --direction_weight 0.3
"""

import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os
import json
from datetime import datetime

from models import TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, load_content
from utils.losses import DirectionalLoss

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def calculate_trading_pnl(predictions, actuals, prev_values, 
                          initial_capital=100000000, transaction_cost=0.001):
    """
    Calculate P&L from simulated trading based on predictions.
    
    Args:
        predictions: numpy array of shape (n_samples, pred_len, features)
        actuals: numpy array of shape (n_samples, pred_len, features)
        prev_values: numpy array of shape (n_samples,) - previous actual values
        initial_capital: Starting capital in VND
        transaction_cost: Transaction cost as percentage (default 0.1%)
    
    Returns:
        Dictionary with trading metrics
    """
    capital = float(initial_capital)
    position = 0  # 0: no position, 1: long
    shares = 0
    entry_price = 0
    
    trades = []
    capital_history = [capital]
    
    threshold = 0.002  # 0.2% threshold to trigger trade
    
    n_samples = len(predictions)
    
    for i in range(n_samples):
        prev_price = float(prev_values[i])
        pred_price = float(predictions[i, 0, 0]) if len(predictions.shape) == 3 else float(predictions[i, 0])
        actual_price = float(actuals[i, 0, 0]) if len(actuals.shape) == 3 else float(actuals[i, 0])
        
        # Skip if prices are invalid
        if prev_price <= 0 or actual_price <= 0:
            capital_history.append(capital_history[-1])
            continue
        
        pred_change = (pred_price - prev_price) / prev_price
        current_price = actual_price
        
        # BUY signal: predicted to go up
        if pred_change > threshold and position == 0:
            shares = int((capital * 0.95) / current_price)
            if shares > 0:
                cost = shares * current_price * (1 + transaction_cost)
                if cost <= capital:
                    capital -= cost
                    entry_price = current_price
                    position = 1
                    trades.append({'type': 'buy', 'price': current_price, 'shares': shares})
        
        # SELL signal: predicted to go down
        elif pred_change < -threshold and position == 1:
            revenue = shares * current_price * (1 - transaction_cost)
            pnl = revenue - (entry_price * shares)
            capital += revenue
            trades.append({'type': 'sell', 'price': current_price, 'shares': shares, 'pnl': pnl})
            position = 0
            shares = 0
            entry_price = 0
        
        # Calculate current portfolio value
        if position == 1:
            total_value = capital + shares * actual_price
        else:
            total_value = capital
        
        capital_history.append(float(total_value))
    
    # Close any remaining position at the end
    if position == 1 and len(actuals) > 0:
        final_price = float(actuals[-1, 0, 0]) if len(actuals.shape) == 3 else float(actuals[-1, 0])
        revenue = shares * final_price * (1 - transaction_cost)
        pnl = revenue - (entry_price * shares)
        capital += revenue
        trades.append({'type': 'sell_final', 'price': final_price, 'shares': shares, 'pnl': pnl})
        position = 0
        shares = 0
    
    final_capital = float(capital)
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # Buy and hold return
    if n_samples > 0:
        first_price = float(prev_values[0])
        last_price = float(actuals[-1, 0, 0]) if len(actuals.shape) == 3 else float(actuals[-1, 0])
        buy_hold_return = (last_price - first_price) / first_price * 100 if first_price > 0 else 0
    else:
        buy_hold_return = 0
    
    # Trade statistics
    winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    total_closed_trades = sum(1 for t in trades if 'pnl' in t)
    trade_win_rate = winning_trades / (total_closed_trades + 1e-8) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_capital': float(final_capital),
        'total_return_pct': float(total_return),
        'buy_hold_return_pct': float(buy_hold_return),
        'excess_return_pct': float(total_return - buy_hold_return),
        'total_trades': int(total_closed_trades),
        'winning_trades': int(winning_trades),
        'trade_win_rate_pct': float(trade_win_rate),
    }


def vali_with_metrics(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric,
                      compute_trading=True):
    """
    Validation with comprehensive metrics including winrate and P&L.
    
    Returns:
        total_loss: Average MSE loss
        total_mae_loss: Average MAE loss
        direction_acc: Directional accuracy (winrate) percentage
        trading_metrics: Dictionary with P&L metrics (if compute_trading=True)
    """
    total_loss = []
    total_mae_loss = []
    correct_directions = 0
    total_samples = 0
    
    # For trading calculation
    all_predictions = []
    all_actuals = []
    all_prev_values = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(vali_loader, desc="Validating", leave=False):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
            
            # Get previous values for direction calculation
            prev_values = batch_x[:, -1, f_dim].to(accelerator.device)
            
            pred = outputs.detach()
            true = batch_y.detach()
            
            # MSE loss (use base MSE for comparison, not directional)
            if hasattr(criterion, 'mse'):
                loss = criterion.mse(pred, true)
            else:
                loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
            # Calculate directional accuracy
            pred_dir = (pred[:, 0, 0] > prev_values).float()
            true_dir = (true[:, 0, 0] > prev_values).float()
            correct_directions += (pred_dir == true_dir).sum().item()
            total_samples += pred.shape[0]
            
            # Store for trading calculation
            if compute_trading:
                all_predictions.append(pred.cpu().numpy())
                all_actuals.append(true.cpu().numpy())
                all_prev_values.append(prev_values.cpu().numpy())
    
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    direction_acc = correct_directions / total_samples * 100 if total_samples > 0 else 0
    
    # Calculate trading metrics
    trading_metrics = None
    if compute_trading and len(all_predictions) > 0:
        predictions = np.concatenate(all_predictions, axis=0)
        actuals = np.concatenate(all_actuals, axis=0)
        prev_values = np.concatenate(all_prev_values, axis=0)
        trading_metrics = calculate_trading_pnl(predictions, actuals, prev_values)
    
    model.train()
    return total_loss, total_mae_loss, direction_acc, trading_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='TimeLLM with Directional Loss')
    
    # ===== Directional Loss Options =====
    parser.add_argument('--use_directional_loss', type=int, default=0, choices=[0, 1],
                        help='0: MSE only (baseline), 1: MSE + Directional Loss')
    parser.add_argument('--direction_weight', type=float, default=0.3,
                        help='Weight for directional loss component (only used if use_directional_loss=1)')
    parser.add_argument('--use_soft_direction', type=int, default=1,
                        help='Use soft (differentiable) directional loss')
    
    # ===== Patching Mode =====
    parser.add_argument('--patching_mode', type=str, default='frequency_aware',
                        choices=['frequency_aware', 'multi_scale', 'single'],
                        help='Patching mode: frequency_aware (FFT+attention), multi_scale, single')
    
    # ===== Basic Config =====
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='Weather_96_96')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-FFT')
    parser.add_argument('--model', type=str, default='TimeLLM')
    parser.add_argument('--seed', type=int, default=2021)
    
    # ===== Data Config =====
    parser.add_argument('--data', type=str, default='Weather')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/weather/')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='M:multivariate, S:univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--loader', type=str, default='modal')
    
    # ===== Forecasting Config =====
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    
    # ===== Model Config =====
    parser.add_argument('--enc_in', type=int, default=21)
    parser.add_argument('--dec_in', type=int, default=21)
    parser.add_argument('--c_out', type=int, default=21)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--prompt_domain', type=int, default=0)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)
    
    # ===== Training Config =====
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--percent', type=int, default=100)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    try:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    except:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # Create experiment name based on settings
    loss_type = f"DirectionalLoss_w{args.direction_weight}" if args.use_directional_loss else "MSE"
    
    for ii in range(args.itr):
        # Setting string for checkpoint naming
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_{}_pm{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            loss_type,
            args.patching_mode
        )
        
        # Load data
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        
        # Load content/prompt
        args.content = load_content(args)
        
        # Initialize model
        model = TimeLLM.Model(args).float()
        
        accelerator.print(f"\n{'='*70}")
        accelerator.print(f"Experiment: {setting}")
        accelerator.print(f"{'='*70}")
        accelerator.print(f"  Model: {args.model}")
        accelerator.print(f"  Patching Mode: {args.patching_mode}")
        accelerator.print(f"  Loss: {loss_type}")
        accelerator.print(f"  Directional Loss: {'Enabled' if args.use_directional_loss else 'Disabled'}")
        if args.use_directional_loss:
            accelerator.print(f"  Direction Weight: {args.direction_weight}")
        accelerator.print(f"{'='*70}\n")
        
        # Checkpoint path
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
        
        # Get trainable parameters
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
        
        # Scheduler
        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=args.pct_start,
                epochs=args.train_epochs,
                max_lr=args.learning_rate
            )
        
        # Loss function
        if args.use_directional_loss:
            criterion = DirectionalLoss(
                direction_weight=args.direction_weight,
                use_soft_direction=bool(args.use_soft_direction)
            )
        else:
            criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        
        # Prepare with accelerator
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)
        
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # Training history
        history = {
            'train_loss': [], 'vali_loss': [], 'test_loss': [],
            'train_mae': [], 'vali_mae': [], 'test_mae': [],
            'vali_winrate': [], 'test_winrate': [],
            'vali_pnl': [], 'test_pnl': [],
            'vali_excess_return': [], 'test_excess_return': []
        }
        
        best_vali_loss = float('inf')
        best_vali_winrate = 0
        
        # Training loop
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            train_correct = 0
            train_total = 0
            
            model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"
            ):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
                
                # Forward pass
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
                
                # Compute loss
                if args.use_directional_loss:
                    prev_values = batch_x[:, -1, f_dim]
                    loss = criterion(outputs, batch_y_target, prev_values)
                    
                    # Track directional accuracy
                    with torch.no_grad():
                        pred_dir = (outputs[:, 0, 0] > prev_values).float()
                        true_dir = (batch_y_target[:, 0, 0] > prev_values).float()
                        train_correct += (pred_dir == true_dir).sum().item()
                        train_total += outputs.shape[0]
                else:
                    loss = criterion(outputs, batch_y_target)
                
                train_loss.append(loss.item())
                
                # Backward pass
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()
                
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
            
            train_loss_avg = np.average(train_loss)
            train_winrate = train_correct / train_total * 100 if train_total > 0 else 0
            
            epoch_duration = time.time() - epoch_time
            accelerator.print(f"\nEpoch {epoch+1} | Time: {epoch_duration:.2f}s | Train Loss: {train_loss_avg:.6f}", end="")
            if args.use_directional_loss:
                accelerator.print(f" | Train Winrate: {train_winrate:.2f}%")
            else:
                accelerator.print("")
            
            # Validation
            vali_loss, vali_mae, vali_winrate, vali_trading = vali_with_metrics(
                args, accelerator, model, vali_data, vali_loader, criterion, mae_metric,
                compute_trading=True
            )
            
            # Test
            test_loss, test_mae, test_winrate, test_trading = vali_with_metrics(
                args, accelerator, model, test_data, test_loader, criterion, mae_metric,
                compute_trading=True
            )
            
            # Log results
            accelerator.print(f"  Vali Loss: {vali_loss:.6f} | Vali MAE: {vali_mae:.6f} | Vali Winrate: {vali_winrate:.2f}%")
            if vali_trading:
                accelerator.print(f"  Vali P&L: {vali_trading['total_return_pct']:+.2f}% | Excess: {vali_trading['excess_return_pct']:+.2f}%")
            
            accelerator.print(f"  Test Loss: {test_loss:.6f} | Test MAE: {test_mae:.6f} | Test Winrate: {test_winrate:.2f}%")
            if test_trading:
                accelerator.print(f"  Test P&L: {test_trading['total_return_pct']:+.2f}% | Excess: {test_trading['excess_return_pct']:+.2f}%")
            
            # Update history
            history['train_loss'].append(float(train_loss_avg))
            history['vali_loss'].append(float(vali_loss))
            history['test_loss'].append(float(test_loss))
            history['train_mae'].append(float(train_loss_avg))  # Approximate
            history['vali_mae'].append(float(vali_mae))
            history['test_mae'].append(float(test_mae))
            history['vali_winrate'].append(float(vali_winrate))
            history['test_winrate'].append(float(test_winrate))
            
            if vali_trading:
                history['vali_pnl'].append(float(vali_trading['total_return_pct']))
                history['vali_excess_return'].append(float(vali_trading['excess_return_pct']))
            if test_trading:
                history['test_pnl'].append(float(test_trading['total_return_pct']))
                history['test_excess_return'].append(float(test_trading['excess_return_pct']))
            
            # Track best
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
            if vali_winrate > best_vali_winrate:
                best_vali_winrate = vali_winrate
                accelerator.print(f"  >> New best validation winrate: {best_vali_winrate:.2f}%")
            
            # Print patch info for frequency_aware mode
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'print_patch_info'):
                    unwrapped_model.print_patch_info(epoch=epoch + 1)
            
            # Early stopping
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping triggered")
                break
            
            # Learning rate adjustment
            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        
        # Save training history and config
        if accelerator.is_local_main_process:
            # Save history
            history_file = os.path.join(path, 'training_history.json')
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Save config
            config_dict = {k: v for k, v in vars(args).items() if k != 'content'}
            config_dict['experiment_name'] = setting
            config_dict['loss_type'] = loss_type
            config_file = os.path.join(path, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Print final summary
            accelerator.print(f"\n{'='*70}")
            accelerator.print(f"TRAINING COMPLETED - FINAL RESULTS")
            accelerator.print(f"{'='*70}")
            accelerator.print(f"Loss Type: {loss_type}")
            accelerator.print(f"")
            accelerator.print(f"--- VALIDATION SET ---")
            accelerator.print(f"  Winrate: {history['vali_winrate'][-1]:.2f}%")
            accelerator.print(f"  Best Winrate: {best_vali_winrate:.2f}%")
            if history['vali_pnl']:
                accelerator.print(f"  P&L: {history['vali_pnl'][-1]:+.2f}%")
                accelerator.print(f"  Excess Return: {history['vali_excess_return'][-1]:+.2f}%")
            accelerator.print(f"  Loss: {history['vali_loss'][-1]:.6f}")
            accelerator.print(f"")
            accelerator.print(f"--- TEST SET ---")
            accelerator.print(f"  Winrate: {history['test_winrate'][-1]:.2f}%")
            if history['test_pnl']:
                accelerator.print(f"  P&L: {history['test_pnl'][-1]:+.2f}%")
                accelerator.print(f"  Excess Return: {history['test_excess_return'][-1]:+.2f}%")
            accelerator.print(f"  Loss: {history['test_loss'][-1]:.6f}")
            accelerator.print(f"")
            accelerator.print(f"Model saved to: {path}")
            accelerator.print(f"{'='*70}\n")
    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

