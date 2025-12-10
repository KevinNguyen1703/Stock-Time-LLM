"""
Stock Model Evaluation Script
Evaluates Time-LLM model predictions for stock trading performance:
- Win Rate: Percentage of correct direction predictions
- P&L (Profit & Loss): Simulated trading returns
- Visualization: Actual vs Predicted prices
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from datetime import datetime

from models import TimeLLM
from models import TimeLLM_Stock
from data_provider.data_factory import data_provider
from utils.tools import load_content


class StockEvaluator:
    """Evaluate stock prediction model performance"""
    
    def __init__(self, model, args, device='cuda'):
        self.model = model
        self.args = args
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader, data_set, with_prompts=False):
        """
        Generate predictions for all samples in data loader
        
        Returns:
            predictions: numpy array of predicted values
            actuals: numpy array of actual values
        """
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                # Handle batch with or without prompts
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, prompts = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    prompts = None
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                if prompts is not None and hasattr(self.model, 'forecast'):
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dynamic_prompts=prompts)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y_target.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Inverse transform if scaler is available
        if hasattr(data_set, 'scaler') and data_set.scaler is not None:
            pred_shape = predictions.shape
            act_shape = actuals.shape
            
            n_features = data_set.scaler.n_features_in_
            
            # For predictions (only target column)
            pred_full = np.zeros((pred_shape[0] * pred_shape[1], n_features))
            pred_full[:, -1] = predictions.reshape(-1)
            pred_inverse = data_set.scaler.inverse_transform(pred_full)[:, -1]
            predictions = pred_inverse.reshape(pred_shape[0], pred_shape[1], 1)
            
            # For actuals
            act_full = np.zeros((act_shape[0] * act_shape[1], n_features))
            act_full[:, -1] = actuals.reshape(-1)
            act_inverse = data_set.scaler.inverse_transform(act_full)[:, -1]
            actuals = act_inverse.reshape(act_shape[0], act_shape[1], 1)
        
        return predictions, actuals
    
    def calculate_metrics(self, predictions, actuals):
        """Calculate various evaluation metrics"""
        pred_flat = predictions.flatten()
        act_flat = actuals.flatten()
        
        mse = np.mean((pred_flat - act_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_flat - act_flat))
        mape = np.mean(np.abs((act_flat - pred_flat) / (act_flat + 1e-8))) * 100
        
        ss_res = np.sum((act_flat - pred_flat) ** 2)
        ss_tot = np.sum((act_flat - np.mean(act_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2)
        }
    
    def calculate_directional_accuracy(self, predictions, actuals):
        """Calculate directional accuracy (win rate)"""
        n_samples = len(predictions)
        
        # For short-term (pred_len=1), compare with previous actual
        if predictions.shape[1] == 1:
            # Use diff between consecutive predictions as proxy
            pred_direction = np.zeros(n_samples)
            act_direction = np.zeros(n_samples)
            
            for i in range(1, n_samples):
                pred_direction[i] = np.sign(predictions[i, 0, 0] - actuals[i-1, 0, 0])
                act_direction[i] = np.sign(actuals[i, 0, 0] - actuals[i-1, 0, 0])
            
            # Skip first sample
            pred_direction = pred_direction[1:]
            act_direction = act_direction[1:]
            n_samples = len(pred_direction)
        else:
            # For multi-step, compare first and last
            pred_direction = np.sign(predictions[:, -1, 0] - predictions[:, 0, 0])
            act_direction = np.sign(actuals[:, -1, 0] - actuals[:, 0, 0])
        
        correct_predictions = (pred_direction == act_direction).sum()
        accuracy = correct_predictions / n_samples * 100
        
        up_actual = (act_direction > 0).sum()
        down_actual = (act_direction < 0).sum()
        up_pred = (pred_direction > 0).sum()
        down_pred = (pred_direction < 0).sum()
        
        up_correct = ((pred_direction > 0) & (act_direction > 0)).sum()
        down_correct = ((pred_direction < 0) & (act_direction < 0)).sum()
        
        return {
            'directional_accuracy': float(accuracy),
            'correct_predictions': int(correct_predictions),
            'total_predictions': int(n_samples),
            'up_actual': int(up_actual),
            'down_actual': int(down_actual),
            'up_predicted': int(up_pred),
            'down_predicted': int(down_pred),
            'up_correct': int(up_correct),
            'down_correct': int(down_correct),
            'up_precision': float(up_correct / (up_pred + 1e-8) * 100),
            'down_precision': float(down_correct / (down_pred + 1e-8) * 100)
        }
    
    def calculate_trading_pnl(self, predictions, actuals, initial_capital=100000000,
                             transaction_cost=0.001):
        """Calculate P&L from simulated trading"""
        capital = float(initial_capital)
        position = 0  # 0: no position, 1: long
        shares = 0
        entry_price = 0
        
        trades = []
        capital_history = [capital]
        
        threshold = 0.005  # 0.5% threshold to trigger trade
        
        for i in range(1, len(predictions)):
            prev_price = float(actuals[i-1, 0, 0])
            pred_price = float(predictions[i, 0, 0])
            actual_price = float(actuals[i, 0, 0])
            
            # Skip if prices are invalid
            if prev_price <= 0 or actual_price <= 0:
                capital_history.append(capital_history[-1])
                continue
            
            pred_change = (pred_price - prev_price) / prev_price
            current_price = actual_price  # Use actual price for execution
            
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
            final_price = float(actuals[-1, 0, 0])
            revenue = shares * final_price * (1 - transaction_cost)
            pnl = revenue - (entry_price * shares)
            capital += revenue
            trades.append({'type': 'sell_final', 'price': final_price, 'shares': shares, 'pnl': pnl})
            position = 0
            shares = 0
        
        # Calculate final capital (should be just cash now)
        final_capital = float(capital)
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        if len(actuals) > 1:
            buy_hold_return = (actuals[-1, 0, 0] - actuals[0, 0, 0]) / actuals[0, 0, 0] * 100
        else:
            buy_hold_return = 0
        
        # Calculate risk metrics
        capital_array = np.array(capital_history)
        returns = np.diff(capital_array) / (capital_array[:-1] + 1e-8)
        
        # Sharpe ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(capital_array)
        drawdown = (peak - capital_array) / (peak + 1e-8)
        max_drawdown = float(np.max(drawdown) * 100)
        
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
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown_pct': max_drawdown,
            'total_trades': int(total_closed_trades),
            'winning_trades': int(winning_trades),
            'trade_win_rate_pct': float(trade_win_rate),
            'capital_history': [float(c) for c in capital_history]
        }


def load_model(checkpoint_path, args, device='cuda'):
    """Load trained model from checkpoint"""
    args.content = load_content(args)
    
    # Initialize model based on type
    if args.model == 'TimeLLM_Stock':
        model = TimeLLM_Stock.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint')
    if os.path.exists(checkpoint_file):
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        print(f"Loaded model from {checkpoint_file}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    model = model.to(device)
    model.eval()
    
    return model


def plot_predictions(predictions, actuals, output_path, title="Actual vs Predicted Prices", 
                     num_samples=200):
    """Plot actual vs predicted prices"""
    plt.figure(figsize=(15, 10))
    
    # Flatten for plotting (use first prediction step if multi-step)
    pred_plot = predictions[:num_samples, 0, 0]
    act_plot = actuals[:num_samples, 0, 0]
    
    # Plot 1: Time series comparison
    plt.subplot(2, 2, 1)
    plt.plot(act_plot, label='Actual', color='blue', linewidth=1.5)
    plt.plot(pred_plot, label='Predicted', color='red', alpha=0.7, linewidth=1.5)
    plt.title(f'{title} - Time Series')
    plt.xlabel('Sample Index')
    plt.ylabel('Price (VND)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(act_plot, pred_plot, alpha=0.5, s=20)
    min_val = min(act_plot.min(), pred_plot.min())
    max_val = max(act_plot.max(), pred_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.title('Actual vs Predicted (Scatter)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Prediction error distribution
    plt.subplot(2, 2, 3)
    errors = pred_plot - act_plot
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Percentage error over time
    plt.subplot(2, 2, 4)
    pct_errors = (pred_plot - act_plot) / act_plot * 100
    plt.plot(pct_errors, color='green', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Percentage Error Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Error (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction plot to: {output_path}")


def plot_trading_performance(trading_results, output_path):
    """Plot trading performance"""
    capital_history = trading_results['capital_history']
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(capital_history, color='green', linewidth=1.5)
    plt.axhline(y=trading_results['initial_capital'], color='r', linestyle='--', 
                label='Initial Capital')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value (VND)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format y-axis with millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.subplot(1, 2, 2)
    returns = np.diff(capital_history) / np.array(capital_history[:-1]) * 100
    plt.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Daily Return Distribution')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trading performance plot to: {output_path}")


def print_results(results):
    """Print evaluation results"""
    for split in ['train', 'test']:
        if split not in results:
            continue
            
        print(f"\n{'='*60}")
        print(f" {split.upper()} SET RESULTS")
        print(f"{'='*60}")
        
        metrics = results[split]['metrics']
        print("\n📊 Prediction Metrics:")
        print(f"   MSE:  {metrics['MSE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE:  {metrics['MAE']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   R²:   {metrics['R2']:.4f}")
        
        dir_acc = results[split]['directional']
        print("\n🎯 Directional Accuracy (Win Rate):")
        print(f"   Overall Accuracy: {dir_acc['directional_accuracy']:.2f}%")
        print(f"   Correct/Total: {dir_acc['correct_predictions']}/{dir_acc['total_predictions']}")
        print(f"   Up Precision:   {dir_acc['up_precision']:.2f}%")
        print(f"   Down Precision: {dir_acc['down_precision']:.2f}%")
        
        trading = results[split]['trading']
        print("\n💰 Trading Performance:")
        print(f"   Initial Capital:    {trading['initial_capital']:,.0f} VND")
        print(f"   Final Capital:      {trading['final_capital']:,.0f} VND")
        print(f"   Total Return:       {trading['total_return_pct']:+.2f}%")
        print(f"   Buy & Hold Return:  {trading['buy_hold_return_pct']:+.2f}%")
        print(f"   Excess Return:      {trading['excess_return_pct']:+.2f}%")
        print(f"   Sharpe Ratio:       {trading['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown:       {trading['max_drawdown_pct']:.2f}%")
        print(f"   Trade Win Rate:     {trading['trade_win_rate_pct']:.2f}%")
        print(f"   Total Trades:       {trading['total_trades']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stock Prediction Model')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--prediction_type', type=str, default='short_term',
                        choices=['short_term', 'mid_term'])
    parser.add_argument('--use_dynamic_prompt', action='store_true', default=False,
                        help='Use dynamic prompts during evaluation')
    parser.add_argument('--version', type=str, default=None, choices=['v0', 'v1', 'v2'],
                        help='Auto-configure for specific model version (v0/v1/v2)')
    
    # Data config
    parser.add_argument('--data', type=str, default='Stock')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/stock/')
    parser.add_argument('--data_path', type=str, default='vcb_stock_indicators.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='Adj Close')
    parser.add_argument('--freq', type=str, default='d')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='TimeLLM_Stock',
                        choices=['TimeLLM', 'TimeLLM_Stock'])
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--label_len', type=int, default=30)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--enc_in', type=int, default=6)
    parser.add_argument('--dec_in', type=int, default=6)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--prompt_domain', type=int, default=1)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)
    
    # Other
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--prompt_data_path', type=str, default=None)
    
    args = parser.parse_args()
    
    # Auto-configure for specific version
    if args.version == 'v0':
        # V0: Basic data, static prompt, original TimeLLM
        print("Configuring for V0 (baseline)...")
        args.model = 'TimeLLM'
        args.data_path = 'vcb_stock_indicators_v0.csv'
        args.enc_in = 5
        args.dec_in = 5
        args.patch_len = 16
        args.stride = 8
        args.use_dynamic_prompt = False
    elif args.version == 'v1':
        # V1: V2 data (with momentum), ChatGPT prompts, TimeLLM_Stock
        print("Configuring for V1 (momentum + ChatGPT prompts)...")
        args.model = 'TimeLLM_Stock'
        args.data_path = 'vcb_stock_indicators_v2.csv'  # Uses V2 data with momentum
        args.enc_in = 13
        args.dec_in = 13
        args.patch_len = 16
        args.stride = 8
        args.use_dynamic_prompt = True
    elif args.version == 'v2':
        # V2: Enhanced data, generated prompts, TimeLLM_Stock
        print("Configuring for V2 (enhanced data + prompts)...")
        args.model = 'TimeLLM_Stock'
        args.data_path = 'vcb_stock_indicators_v2.csv'
        args.enc_in = 13
        args.dec_in = 13
        args.patch_len = 16
        args.stride = 8
        args.use_dynamic_prompt = True
    
    # Set prediction length
    if args.prediction_type == 'short_term':
        args.pred_len = 1
        if args.prompt_data_path is None:
            if args.version == 'v2':
                args.prompt_data_path = 'prompts_v2_short_term.json'
            else:
                args.prompt_data_path = 'prompts_short_term.json'
    else:
        args.pred_len = 60
        if args.prompt_data_path is None:
            if args.version == 'v2':
                args.prompt_data_path = 'prompts_v2_mid_term.json'
            else:
                args.prompt_data_path = 'prompts_mid_term.json'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint_path}")
    model = load_model(args.checkpoint_path, args, device)
    
    # Load data
    print(f"Loading data with prompts: {args.use_dynamic_prompt}")
    train_data, train_loader = data_provider(args, 'train', with_prompt=args.use_dynamic_prompt)
    test_data, test_loader = data_provider(args, 'test', with_prompt=args.use_dynamic_prompt)
    
    # Initialize evaluator
    evaluator = StockEvaluator(model, args, device)
    
    results = {}
    
    # Evaluate on training data
    print("\n--- Evaluating on Training Data ---")
    train_preds, train_actuals = evaluator.predict(train_loader, train_data, args.use_dynamic_prompt)
    results['train'] = {
        'metrics': evaluator.calculate_metrics(train_preds, train_actuals),
        'directional': evaluator.calculate_directional_accuracy(train_preds, train_actuals),
        'trading': evaluator.calculate_trading_pnl(train_preds, train_actuals)
    }
    
    # Evaluate on test data
    print("\n--- Evaluating on Test Data ---")
    test_preds, test_actuals = evaluator.predict(test_loader, test_data, args.use_dynamic_prompt)
    results['test'] = {
        'metrics': evaluator.calculate_metrics(test_preds, test_actuals),
        'directional': evaluator.calculate_directional_accuracy(test_preds, test_actuals),
        'trading': evaluator.calculate_trading_pnl(test_preds, test_actuals)
    }
    
    # Print results
    print_results(results)
    
    # Save results and plots
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    results_file = os.path.join(args.output_dir, f'evaluation_{args.prediction_type}_{timestamp}.json')
    results_save = {
        split: {
            'metrics': results[split]['metrics'],
            'directional': results[split]['directional'],
            'trading': {k: v for k, v in results[split]['trading'].items() if k != 'capital_history'}
        }
        for split in results
    }
    with open(results_file, 'w') as f:
        json.dump(results_save, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save predictions
    np.savez(
        os.path.join(args.output_dir, f'predictions_{args.prediction_type}_{timestamp}.npz'),
        train_predictions=train_preds,
        train_actuals=train_actuals,
        test_predictions=test_preds,
        test_actuals=test_actuals
    )
    
    # Generate plots
    plot_predictions(
        train_preds, train_actuals,
        os.path.join(args.output_dir, f'train_predictions_{timestamp}.png'),
        title="Training Set - Actual vs Predicted"
    )
    
    plot_predictions(
        test_preds, test_actuals,
        os.path.join(args.output_dir, f'test_predictions_{timestamp}.png'),
        title="Test Set - Actual vs Predicted"
    )
    
    plot_trading_performance(
        results['train']['trading'],
        os.path.join(args.output_dir, f'train_trading_{timestamp}.png')
    )
    
    plot_trading_performance(
        results['test']['trading'],
        os.path.join(args.output_dir, f'test_trading_{timestamp}.png')
    )
    
    print(f"\nAll results and plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
