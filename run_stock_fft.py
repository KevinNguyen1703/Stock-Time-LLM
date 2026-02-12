"""
Stock Training Script with FFT+Attention Patching
==================================================
Based on run_main.py with added patching_mode support.
Outputs: MSE, MAE loss for train, valid, test sets.

Usage:
    python run_stock_fft.py --patching_mode frequency_aware ...
"""

import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content


def parse_args():
    parser = argparse.ArgumentParser(description='Stock Training with FFT+Attention')
    
    # Patching mode
    parser.add_argument('--patching_mode', type=str, default='frequency_aware',
                        choices=['frequency_aware', 'multi_scale', 'single'],
                        help='Patching mode: frequency_aware (FFT+attention), multi_scale, single')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='VCB_stock_60_1')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-FFT')
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
    parser.add_argument('--loader', type=str, default='modal')
    
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
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--percent', type=int, default=100)
    
    return parser.parse_args()


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
    
    for ii in range(args.itr):
        # Setting string for checkpoint naming
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_pm{}'.format(
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
        
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"Stock Training with FFT+Attention")
        accelerator.print(f"{'='*60}")
        accelerator.print(f"  Patching Mode: {args.patching_mode}")
        accelerator.print(f"  Data: {args.data_path}")
        accelerator.print(f"  Seq Len: {args.seq_len} -> Pred Len: {args.pred_len}")
        accelerator.print(f"  Features: {args.enc_in}")
        accelerator.print(f"{'='*60}\n")
        
        # Checkpoint path
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)
        
        time_now = time.time()
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
        
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        
        # Prepare with accelerator
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)
        
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mae = []
            
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
                
                loss = criterion(outputs, batch_y_target)
                mae_loss = mae_metric(outputs, batch_y_target)
                
                train_loss.append(loss.item())
                train_mae.append(mae_loss.item())
                
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
            
            # Epoch metrics
            train_mse = np.average(train_loss)
            train_mae_avg = np.average(train_mae)
            
            # Validation and Test
            vali_mse, vali_mae = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_mse, test_mae = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            
            epoch_time_cost = time.time() - epoch_time
            
            # Print results
            accelerator.print(f"\nEpoch {epoch+1} | Time: {epoch_time_cost:.2f}s")
            accelerator.print(f"  Train MSE: {train_mse:.6f} | Train MAE: {train_mae_avg:.6f}")
            accelerator.print(f"  Valid MSE: {vali_mse:.6f} | Valid MAE: {vali_mae:.6f}")
            accelerator.print(f"  Test  MSE: {test_mse:.6f} | Test  MAE: {test_mae:.6f}")
            
            # Print patch info for frequency_aware mode
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'print_patch_info'):
                    unwrapped_model.print_patch_info(epoch=epoch + 1)
            
            # Early stopping
            early_stopping(vali_mse, model, path)
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
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
        
        # Final summary
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"Training Complete!")
        accelerator.print(f"Model saved to: {path}")
        accelerator.print(f"{'='*60}\n")
    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()



