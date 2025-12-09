"""
Stock Price Prediction Training V2 - Improved for Directional Accuracy
Includes:
- Directional loss component
- Better regularization
- More features
"""

import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM_Stock
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
    Combined loss: MSE + Directional penalty
    Penalizes predictions that get the direction wrong
    """
    def __init__(self, direction_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
    
    def forward(self, pred, target, prev_values=None):
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        # Directional loss: penalize wrong direction predictions
        # pred and target shape: (batch, pred_len, 1)
        pred_direction = torch.sign(pred[:, 0, 0] - prev_values)
        target_direction = torch.sign(target[:, 0, 0] - prev_values)
        
        # Direction mismatch penalty
        direction_mismatch = (pred_direction != target_direction).float()
        direction_loss = direction_mismatch.mean()
        
        # Combined loss
        total_loss = mse_loss + self.direction_weight * direction_loss
        
        return total_loss


def vali_with_direction(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    """Validation with directional accuracy tracking"""
    total_loss = []
    total_mae_loss = []
    correct_directions = 0
    total_samples = 0
    
    model.eval()
    prev_actual = None
    
    with torch.no_grad():
        for batch in tqdm(vali_loader, desc="Validating"):
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
            
            if prompts is not None:
                unwrapped = accelerator.unwrap_model(model)
                outputs = unwrapped(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
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

            loss = criterion.mse(pred, true)
            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
            # Calculate directional accuracy
            pred_dir = (pred[:, 0, 0] > prev_values).float()
            true_dir = (true[:, 0, 0] > prev_values).float()
            correct_directions += (pred_dir == true_dir).sum().item()
            total_samples += pred.shape[0]

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    direction_acc = correct_directions / total_samples * 100 if total_samples > 0 else 0

    model.train()
    return total_loss, total_mae_loss, direction_acc


def parse_args():
    parser = argparse.ArgumentParser(description='Time-LLM Stock Prediction V2')
    
    parser.add_argument('--prediction_type', type=str, default='short_term',
                        choices=['short_term', 'mid_term'])
    parser.add_argument('--use_dynamic_prompt', action='store_true', default=True)
    parser.add_argument('--prompt_data_path', type=str, default=None)
    parser.add_argument('--use_v2_data', action='store_true', default=True,
                        help='Use V2 dataset with momentum features')
    parser.add_argument('--direction_weight', type=float, default=0.3,
                        help='Weight for directional loss component')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='VCB_stock_v2')
    parser.add_argument('--model_comment', type=str, default='TimeLLM-Stock-V2')
    parser.add_argument('--model', type=str, default='TimeLLM_Stock')
    parser.add_argument('--seed', type=int, default=2021)
    
    # Data config
    parser.add_argument('--data', type=str, default='Stock')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/stock/')
    parser.add_argument('--data_path', type=str, default='vcb_stock_indicators_v2.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='Adj Close')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Forecasting config
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--label_len', type=int, default=30)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    
    # Model config - V2 has more features
    parser.add_argument('--enc_in', type=int, default=13)  # 13 features in V2
    parser.add_argument('--dec_in', type=int, default=13)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)  # Higher dropout
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
    parser.add_argument('--train_epochs', type=int, default=30)  # Fewer epochs
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=7)  # Earlier stopping
    parser.add_argument('--learning_rate', type=float, default=0.0005)  # Lower LR
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--loader', type=str, default='modal')
    
    args = parser.parse_args()
    
    # Set paths based on prediction type
    if args.prediction_type == 'short_term':
        args.pred_len = 1
        args.model_id = f'{args.model_id}_60_1'
        if args.prompt_data_path is None:
            args.prompt_data_path = 'prompts_v2_short_term.json'
    else:
        args.pred_len = 60
        args.model_id = f'{args.model_id}_60_60'
        args.batch_size = 8
        args.learning_rate = 0.0003
        if args.prompt_data_path is None:
            args.prompt_data_path = 'prompts_v2_mid_term.json'
    
    args.model_comment = f'{args.model_comment}-DirectionalLoss'
    
    return args


def train_model_v2(args):
    """Training with directional loss"""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_dw{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.d_ff, args.direction_weight)
    
    # Load content first
    args.content = load_content(args)
    
    # Load data
    accelerator.print(f"\nLoading V2 data: {args.data_path}")
    accelerator.print(f"Dynamic prompts: {args.use_dynamic_prompt}")
    accelerator.print(f"Directional loss weight: {args.direction_weight}")
    
    train_data, train_loader = data_provider(args, 'train', with_prompt=args.use_dynamic_prompt)
    vali_data, vali_loader = data_provider(args, 'val', with_prompt=args.use_dynamic_prompt)
    test_data, test_loader = data_provider(args, 'test', with_prompt=args.use_dynamic_prompt)
    
    # Initialize model
    model = TimeLLM_Stock.Model(args).float()
    accelerator.print(f"Model: {args.model} with {args.enc_in} input features")
    
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)
    
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, weight_decay=1e-5)
    
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate
    )
    
    # Use directional loss
    criterion = DirectionalLoss(direction_weight=args.direction_weight)
    mae_metric = nn.L1Loss()
    
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    training_history = {
        'train_loss': [], 'vali_loss': [], 'test_loss': [],
        'train_dir_acc': [], 'vali_dir_acc': [], 'test_dir_acc': []
    }
    
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Starting V2 Training with Directional Loss")
    accelerator.print(f"{'='*60}\n")
    
    best_dir_acc = 0
    
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        train_correct = 0
        train_total = 0
        
        model.train()
        epoch_time = time.time()
        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
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
            
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # Forward pass
            if prompts is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                outputs = unwrapped_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
            
            # Get previous values for directional loss
            prev_values = batch_x[:, -1, f_dim]
            
            # Compute loss with direction component
            loss = criterion(outputs, batch_y_target, prev_values)
            train_loss.append(loss.item())
            
            # Track directional accuracy
            pred_dir = (outputs[:, 0, 0] > prev_values).float()
            true_dir = (batch_y_target[:, 0, 0] > prev_values).float()
            train_correct += (pred_dir == true_dir).sum().item()
            train_total += outputs.shape[0]
            
            accelerator.backward(loss)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trained_parameters, max_norm=1.0)
            
            model_optim.step()
            scheduler.step()
        
        train_loss_avg = np.average(train_loss)
        train_dir_acc = train_correct / train_total * 100 if train_total > 0 else 0
        
        accelerator.print(f"Epoch {epoch+1} | Time: {time.time()-epoch_time:.2f}s | Train Loss: {train_loss_avg:.6f} | Train Dir Acc: {train_dir_acc:.2f}%")
        
        # Validation
        vali_loss, vali_mae, vali_dir_acc = vali_with_direction(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae, test_dir_acc = vali_with_direction(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        
        accelerator.print(
            f"         | Vali Loss: {vali_loss:.6f} | Vali Dir Acc: {vali_dir_acc:.2f}% | "
            f"Test Loss: {test_loss:.6f} | Test Dir Acc: {test_dir_acc:.2f}%"
        )
        
        training_history['train_loss'].append(float(train_loss_avg))
        training_history['vali_loss'].append(float(vali_loss))
        training_history['test_loss'].append(float(test_loss))
        training_history['train_dir_acc'].append(float(train_dir_acc))
        training_history['vali_dir_acc'].append(float(vali_dir_acc))
        training_history['test_dir_acc'].append(float(test_dir_acc))
        
        # Save best model based on validation directional accuracy
        if vali_dir_acc > best_dir_acc:
            best_dir_acc = vali_dir_acc
            accelerator.print(f"         | New best validation directional accuracy: {best_dir_acc:.2f}%")
        
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break
        
        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
    
    # Save training history
    if accelerator.is_local_main_process:
        with open(os.path.join(path, 'training_history_v2.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save config
        config_dict = {k: v for k, v in vars(args).items() if k != 'content'}
        with open(os.path.join(path, 'config_v2.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    accelerator.wait_for_everyone()
    
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Training completed!")
    accelerator.print(f"Best validation directional accuracy: {best_dir_acc:.2f}%")
    accelerator.print(f"Model saved to: {path}")
    accelerator.print(f"{'='*60}")
    
    return path, training_history


if __name__ == "__main__":
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"Time-LLM Stock Prediction V2")
    print(f"{'='*60}")
    print(f"Using V2 data with momentum features")
    print(f"Directional loss weight: {args.direction_weight}")
    print(f"{'='*60}\n")
    
    checkpoint_path, history = train_model_v2(args)

