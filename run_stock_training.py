"""
Stock Price Prediction Training Script for Time-LLM with Dynamic Prompts
Supports both short-term (1-day) and mid-term (60-day) predictions
With per-sample dynamic prompts (professor advice) during training
"""

import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM
from models import TimeLLM_Stock
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content


def vali_with_prompt(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    """Validation with dynamic prompts"""
    total_loss = []
    total_mae_loss = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(vali_loader, desc="Validating"):
            # Check if prompts are included
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                prompts = None
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            
            # Forward pass with dynamic prompts
            if prompts is not None and hasattr(model, 'module'):
                outputs = model.module(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
            elif prompts is not None:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
            else:
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
    parser = argparse.ArgumentParser(description='Time-LLM Stock Prediction with Dynamic Prompts')
    
    # Prediction type
    parser.add_argument('--prediction_type', type=str, default='short_term',
                        choices=['short_term', 'mid_term'],
                        help='Prediction type: short_term (1 day) or mid_term (60 days)')
    
    # Dynamic prompt settings
    parser.add_argument('--use_dynamic_prompt', action='store_true', default=True,
                        help='Use dynamic per-sample prompts (professor advice)')
    parser.add_argument('--prompt_data_path', type=str, default=None,
                        help='Path to prompt JSON file (auto-detected if not specified)')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=1, help='training status')
    parser.add_argument('--model_id', type=str, default='VCB_stock', help='model id')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-Stock', help='model comment')
    parser.add_argument('--model', type=str, default='TimeLLM_Stock', 
                        choices=['TimeLLM', 'TimeLLM_Stock'],
                        help='model name (TimeLLM_Stock supports dynamic prompts)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    
    # Data config
    parser.add_argument('--data', type=str, default='Stock', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/stock/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='vcb_stock_indicators.csv',
                        help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task: M, S, MS')
    parser.add_argument('--target', type=str, default='Adj Close',
                        help='target feature')
    parser.add_argument('--freq', type=str, default='d', help='time frequency')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='model checkpoints')
    
    # Forecasting config (will be set based on prediction_type)
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=30, help='label length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='seasonal patterns')
    
    # Model config
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attention factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='output attention')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=4, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=1, help='enable domain prompts')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default=768, help='LLM dimension')
    parser.add_argument('--llm_layers', type=int, default=6, help='LLM layers')
    
    # Training config
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--itr', type=int, default=1, help='experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='eval batch size')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use mixed precision')
    parser.add_argument('--percent', type=int, default=100, help='data percentage')
    parser.add_argument('--loader', type=str, default='modal', help='data loader type')
    
    args = parser.parse_args()
    
    # Set prediction length based on type
    if args.prediction_type == 'short_term':
        args.pred_len = 1
        args.model_id = f'{args.model_id}_60_1'
        args.model_comment = f'{args.model_comment}-ShortTerm'
        if args.prompt_data_path is None:
            args.prompt_data_path = 'prompts_short_term.json'
    else:  # mid_term
        args.pred_len = 60
        args.model_id = f'{args.model_id}_60_60'
        args.model_comment = f'{args.model_comment}-MidTerm'
        args.batch_size = 8  # Reduce batch size for longer predictions
        args.learning_rate = 0.0005  # Lower learning rate
        if args.prompt_data_path is None:
            args.prompt_data_path = 'prompts_mid_term.json'
    
    if args.use_dynamic_prompt:
        args.model_comment = f'{args.model_comment}-DynamicPrompt'
    
    return args


def train_model(args):
    """Main training function with dynamic prompt support"""
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # Create setting string
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, 0)
    
    # Load data with prompt support
    accelerator.print(f"\nLoading data with dynamic prompts: {args.use_dynamic_prompt}")
    train_data, train_loader = data_provider(args, 'train', with_prompt=args.use_dynamic_prompt)
    vali_data, vali_loader = data_provider(args, 'val', with_prompt=args.use_dynamic_prompt)
    test_data, test_loader = data_provider(args, 'test', with_prompt=args.use_dynamic_prompt)
    
    # Initialize model
    if args.model == 'TimeLLM_Stock':
        model = TimeLLM_Stock.Model(args).float()
        accelerator.print("Using TimeLLM_Stock model with dynamic prompt support")
    else:
        model = TimeLLM.Model(args).float()
        accelerator.print("Using standard TimeLLM model")
    
    # Create checkpoint path
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    
    # Load content for prompts
    args.content = load_content(args)
    
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)
    
    # Training setup
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    
    # Get trainable parameters
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    
    # Learning rate scheduler
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
    time_now = time.time()
    training_history = {'train_loss': [], 'vali_loss': [], 'test_loss': []}
    
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Starting training with dynamic prompts")
    accelerator.print(f"Prediction type: {args.prediction_type}")
    accelerator.print(f"Seq len: {args.seq_len}, Pred len: {args.pred_len}")
    accelerator.print(f"{'='*60}\n")
    
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        
        model.train()
        epoch_time = time.time()
        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
            # Handle batch with or without prompts
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                prompts = None
            
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # Forward pass with dynamic prompts
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if prompts is not None:
                        # Use unwrapped model for prompt passing
                        unwrapped_model = accelerator.unwrap_model(model)
                        outputs = unwrapped_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y)
            else:
                if prompts is not None:
                    unwrapped_model = accelerator.unwrap_model(model)
                    outputs = unwrapped_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
            
            train_loss.append(loss.item())
            
            if (i + 1) % 100 == 0:
                accelerator.print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                iter_count = 0
                time_now = time.time()
            
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
        
        accelerator.print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")
        train_loss_avg = np.average(train_loss)
        
        # Validation
        vali_loss, vali_mae_loss = vali_with_prompt(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali_with_prompt(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f} "
            f"Test Loss: {test_loss:.7f} MAE Loss: {test_mae_loss:.7f}"
        )
        
        training_history['train_loss'].append(float(train_loss_avg))
        training_history['vali_loss'].append(float(vali_loss))
        training_history['test_loss'].append(float(test_loss))
        
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break
        
        # Learning rate adjustment
        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
    
    # Save training history and config
    if accelerator.is_local_main_process:
        history_path = os.path.join(path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        accelerator.print(f"Training history saved to {history_path}")
        
        # Save config
        config_path = os.path.join(path, 'config.json')
        config_dict = vars(args).copy()
        config_dict.pop('content', None)  # Remove content as it's long
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    accelerator.wait_for_everyone()
    
    return path, training_history


if __name__ == "__main__":
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"Time-LLM Stock Prediction Training")
    print(f"{'='*60}")
    print(f"Prediction type: {args.prediction_type}")
    print(f"Sequence length: {args.seq_len}, Prediction length: {args.pred_len}")
    print(f"Dynamic prompts: {args.use_dynamic_prompt}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")
    
    checkpoint_path, history = train_model(args)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'='*60}")
