"""
TimeLLM Training Script with Directional Loss Support
=======================================================
This script trains the TimeLLM model with FFT+Attention (frequency_aware patching)
and provides options for:
- With/Without directional loss for comparison
- Winrate and Confusion Matrix evaluation on validation and test sets

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

from models import TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, load_content
from utils.losses import DirectionalLoss

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def vali_with_metrics(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    """
    Validation with winrate and confusion matrix.
    
    Confusion Matrix:
        - TP (True Positive): Predicted UP, Actual UP
        - TN (True Negative): Predicted DOWN, Actual DOWN  
        - FP (False Positive): Predicted UP, Actual DOWN
        - FN (False Negative): Predicted DOWN, Actual UP
    
    Returns:
        total_loss: Average MSE loss
        total_mae_loss: Average MAE loss
        metrics: Dictionary with winrate and confusion matrix
    """
    total_loss = []
    total_mae_loss = []
    
    # Confusion matrix counters
    tp = 0  # Predicted UP, Actual UP
    tn = 0  # Predicted DOWN, Actual DOWN
    fp = 0  # Predicted UP, Actual DOWN
    fn = 0  # Predicted DOWN, Actual UP
    
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
            
            # MSE loss
            if hasattr(criterion, 'mse'):
                loss = criterion.mse(pred, true)
            else:
                loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
            # Calculate confusion matrix
            # pred_up = 1 if predicted price > previous price, else 0
            # true_up = 1 if actual price > previous price, else 0
            pred_up = (pred[:, 0, 0] > prev_values)
            true_up = (true[:, 0, 0] > prev_values)
            
            # TP: pred_up=True, true_up=True
            tp += (pred_up & true_up).sum().item()
            # TN: pred_up=False, true_up=False  
            tn += (~pred_up & ~true_up).sum().item()
            # FP: pred_up=True, true_up=False
            fp += (pred_up & ~true_up).sum().item()
            # FN: pred_up=False, true_up=True
            fn += (~pred_up & true_up).sum().item()
    
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    
    # Calculate metrics from confusion matrix
    total = tp + tn + fp + fn
    winrate = (tp + tn) / total * 100 if total > 0 else 0
    
    # Precision and Recall for UP predictions
    precision_up = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall_up = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    
    # Precision and Recall for DOWN predictions
    precision_down = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
    recall_down = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    # F1 scores
    f1_up = 2 * precision_up * recall_up / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0
    f1_down = 2 * precision_down * recall_down / (precision_down + recall_down) if (precision_down + recall_down) > 0 else 0
    
    metrics = {
        'winrate': winrate,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total,
        'precision_up': precision_up,
        'recall_up': recall_up,
        'precision_down': precision_down,
        'recall_down': recall_down,
        'f1_up': f1_up,
        'f1_down': f1_down,
        'actual_up': tp + fn,
        'actual_down': tn + fp,
        'pred_up': tp + fp,
        'pred_down': tn + fn,
    }
    
    model.train()
    return total_loss, total_mae_loss, metrics


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
            'vali_winrate': [], 'test_winrate': [],
            'vali_metrics': [], 'test_metrics': []
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
            vali_loss, vali_mae, vali_metrics = vali_with_metrics(
                args, accelerator, model, vali_data, vali_loader, criterion, mae_metric
            )
            
            # Test
            test_loss, test_mae, test_metrics = vali_with_metrics(
                args, accelerator, model, test_data, test_loader, criterion, mae_metric
            )
            
            # Log results
            accelerator.print(f"  Vali | Loss: {vali_loss:.6f} | Winrate: {vali_metrics['winrate']:.2f}%")
            accelerator.print(f"       | TP:{vali_metrics['tp']} TN:{vali_metrics['tn']} FP:{vali_metrics['fp']} FN:{vali_metrics['fn']}")
            accelerator.print(f"       | Prec_UP:{vali_metrics['precision_up']:.1f}% Rec_UP:{vali_metrics['recall_up']:.1f}% | Prec_DN:{vali_metrics['precision_down']:.1f}% Rec_DN:{vali_metrics['recall_down']:.1f}%")
            
            accelerator.print(f"  Test | Loss: {test_loss:.6f} | Winrate: {test_metrics['winrate']:.2f}%")
            accelerator.print(f"       | TP:{test_metrics['tp']} TN:{test_metrics['tn']} FP:{test_metrics['fp']} FN:{test_metrics['fn']}")
            accelerator.print(f"       | Prec_UP:{test_metrics['precision_up']:.1f}% Rec_UP:{test_metrics['recall_up']:.1f}% | Prec_DN:{test_metrics['precision_down']:.1f}% Rec_DN:{test_metrics['recall_down']:.1f}%")
            
            # Update history
            history['train_loss'].append(float(train_loss_avg))
            history['vali_loss'].append(float(vali_loss))
            history['test_loss'].append(float(test_loss))
            history['vali_winrate'].append(float(vali_metrics['winrate']))
            history['test_winrate'].append(float(test_metrics['winrate']))
            history['vali_metrics'].append(vali_metrics)
            history['test_metrics'].append(test_metrics)
            
            # Track best
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
            if vali_metrics['winrate'] > best_vali_winrate:
                best_vali_winrate = vali_metrics['winrate']
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
            
            # Get final metrics
            final_vali = history['vali_metrics'][-1]
            final_test = history['test_metrics'][-1]
            
            # Print final summary
            accelerator.print(f"\n{'='*70}")
            accelerator.print(f"TRAINING COMPLETED - FINAL RESULTS")
            accelerator.print(f"{'='*70}")
            accelerator.print(f"Loss Type: {loss_type}")
            accelerator.print(f"")
            accelerator.print(f"--- VALIDATION SET ---")
            accelerator.print(f"  Winrate: {final_vali['winrate']:.2f}% (Best: {best_vali_winrate:.2f}%)")
            accelerator.print(f"  Confusion Matrix:")
            accelerator.print(f"                  Actual UP    Actual DOWN")
            accelerator.print(f"    Pred UP         {final_vali['tp']:>5}         {final_vali['fp']:>5}")
            accelerator.print(f"    Pred DOWN       {final_vali['fn']:>5}         {final_vali['tn']:>5}")
            accelerator.print(f"  Precision UP: {final_vali['precision_up']:.2f}% | Recall UP: {final_vali['recall_up']:.2f}%")
            accelerator.print(f"  Precision DOWN: {final_vali['precision_down']:.2f}% | Recall DOWN: {final_vali['recall_down']:.2f}%")
            accelerator.print(f"")
            accelerator.print(f"--- TEST SET ---")
            accelerator.print(f"  Winrate: {final_test['winrate']:.2f}%")
            accelerator.print(f"  Confusion Matrix:")
            accelerator.print(f"                  Actual UP    Actual DOWN")
            accelerator.print(f"    Pred UP         {final_test['tp']:>5}         {final_test['fp']:>5}")
            accelerator.print(f"    Pred DOWN       {final_test['fn']:>5}         {final_test['tn']:>5}")
            accelerator.print(f"  Precision UP: {final_test['precision_up']:.2f}% | Recall UP: {final_test['recall_up']:.2f}%")
            accelerator.print(f"  Precision DOWN: {final_test['precision_down']:.2f}% | Recall DOWN: {final_test['recall_down']:.2f}%")
            accelerator.print(f"")
            accelerator.print(f"Model saved to: {path}")
            accelerator.print(f"{'='*70}\n")
    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
