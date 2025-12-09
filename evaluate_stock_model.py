"""
Stock Model Evaluation Script
Evaluates Time-LLM model predictions for stock trading performance:
- Win Rate: Percentage of correct direction predictions
- P&L (Profit & Loss): Simulated trading returns
- Various accuracy metrics
"""

import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from datetime import datetime

from models import TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content


class StockEvaluator:
    """Evaluate stock prediction model performance"""
    
    def __init__(self, model, args, device='cuda'):
        self.model = model
        self.args = args
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader, data_set):
        """
        Generate predictions for all samples in data loader
        
        Returns:
            predictions: numpy array of predicted values
            actuals: numpy array of actual values
            dates: list of corresponding dates (if available)
        """
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader, desc="Predicting"):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
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
            # Reshape for inverse transform
            pred_shape = predictions.shape
            act_shape = actuals.shape
            
            # Create dummy arrays for inverse transform (scaler expects all features)
            n_features = data_set.scaler.n_features_in_
            
            # For predictions (only target column)
            pred_full = np.zeros((pred_shape[0] * pred_shape[1], n_features))
            pred_full[:, -1] = predictions.reshape(-1)  # Adj Close is last column
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
        # Flatten for overall metrics
        pred_flat = predictions.flatten()
        act_flat = actuals.flatten()
        
        # Basic metrics
        mse = np.mean((pred_flat - act_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_flat - act_flat))
        mape = np.mean(np.abs((act_flat - pred_flat) / (act_flat + 1e-8))) * 100
        
        # R-squared
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
    
    def calculate_directional_accuracy(self, predictions, actuals, prev_prices=None):
        """
        Calculate directional accuracy (win rate)
        
        Args:
            predictions: Predicted prices
            actuals: Actual prices
            prev_prices: Previous day prices (for calculating direction)
        
        Returns:
            Dictionary with directional accuracy metrics
        """
        n_samples = len(predictions)
        
        if prev_prices is None:
            # Use first prediction step's actual as previous price proxy
            # This compares if we correctly predict up/down from start of prediction window
            pred_direction = np.sign(predictions[:, -1, 0] - predictions[:, 0, 0])
            act_direction = np.sign(actuals[:, -1, 0] - actuals[:, 0, 0])
        else:
            pred_direction = np.sign(predictions[:, 0, 0] - prev_prices)
            act_direction = np.sign(actuals[:, 0, 0] - prev_prices)
        
        # Calculate accuracy
        correct_predictions = (pred_direction == act_direction).sum()
        accuracy = correct_predictions / n_samples * 100
        
        # Up/Down breakdown
        up_actual = (act_direction > 0).sum()
        down_actual = (act_direction < 0).sum()
        up_pred = (pred_direction > 0).sum()
        down_pred = (pred_direction < 0).sum()
        
        # True positives for up/down
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
                             transaction_cost=0.001, short_allowed=False):
        """
        Calculate P&L from simulated trading
        
        Trading Strategy:
        - If predicted price increase > threshold: BUY
        - If predicted price decrease > threshold: SELL (or short if allowed)
        - Hold otherwise
        
        Args:
            predictions: Predicted prices
            actuals: Actual prices  
            initial_capital: Starting capital in VND
            transaction_cost: Transaction cost as fraction (0.1%)
            short_allowed: Whether short selling is allowed
        
        Returns:
            Dictionary with P&L metrics
        """
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        shares = 0
        entry_price = 0
        
        trades = []
        capital_history = [capital]
        
        # Trading threshold (predicted change > 0.5% to trigger trade)
        threshold = 0.005
        
        for i in range(len(predictions)):
            # Get predicted and actual price change
            if predictions.shape[1] == 1:
                # Short-term: compare with previous actual
                if i > 0:
                    prev_price = actuals[i-1, 0, 0]
                    pred_change = (predictions[i, 0, 0] - prev_price) / prev_price
                    actual_price = actuals[i, 0, 0]
                else:
                    continue
            else:
                # Mid-term: compare first and last prediction
                prev_price = actuals[i, 0, 0]
                pred_change = (predictions[i, -1, 0] - prev_price) / prev_price
                actual_price = actuals[i, -1, 0]
            
            current_price = prev_price  # Price at decision time
            
            # Trading decision
            if pred_change > threshold and position <= 0:
                # Buy signal
                if position == -1:
                    # Close short position first
                    pnl = (entry_price - current_price) * shares
                    capital += pnl - (current_price * shares * transaction_cost)
                    trades.append({
                        'type': 'close_short',
                        'price': current_price,
                        'shares': shares,
                        'pnl': pnl
                    })
                
                # Open long position
                shares = int(capital * 0.95 / current_price)  # Use 95% of capital
                cost = shares * current_price * (1 + transaction_cost)
                if cost <= capital:
                    capital -= cost
                    entry_price = current_price
                    position = 1
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares
                    })
            
            elif pred_change < -threshold and position >= 0:
                # Sell signal
                if position == 1:
                    # Close long position
                    revenue = shares * current_price * (1 - transaction_cost)
                    pnl = revenue - (entry_price * shares)
                    capital += revenue
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'shares': shares,
                        'pnl': pnl
                    })
                    position = 0
                    shares = 0
                
                if short_allowed and position == 0:
                    # Open short position
                    shares = int(capital * 0.95 / current_price)
                    entry_price = current_price
                    position = -1
                    trades.append({
                        'type': 'short',
                        'price': current_price,
                        'shares': shares
                    })
            
            # Update capital history with position value
            if position == 1:
                total_value = capital + shares * actual_price
            elif position == -1:
                total_value = capital + (entry_price - actual_price) * shares
            else:
                total_value = capital
            
            capital_history.append(total_value)
        
        # Close any remaining position
        if position == 1 and len(actuals) > 0:
            final_price = actuals[-1, -1, 0]
            revenue = shares * final_price * (1 - transaction_cost)
            capital += revenue
        elif position == -1 and len(actuals) > 0:
            final_price = actuals[-1, -1, 0]
            pnl = (entry_price - final_price) * shares
            capital += pnl
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        
        # Buy and hold comparison
        if len(actuals) > 1:
            buy_hold_return = (actuals[-1, -1, 0] - actuals[0, 0, 0]) / actuals[0, 0, 0] * 100
        else:
            buy_hold_return = 0
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(capital_history) / np.array(capital_history[:-1])
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        
        # Maximum drawdown
        peak = np.maximum.accumulate(capital_history)
        drawdown = (peak - capital_history) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # Win rate for trades
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total_trades = sum(1 for t in trades if 'pnl' in t)
        trade_win_rate = winning_trades / (total_trades + 1e-8) * 100
        
        return {
            'initial_capital': initial_capital,
            'final_capital': float(capital),
            'total_return_pct': float(total_return),
            'buy_hold_return_pct': float(buy_hold_return),
            'excess_return_pct': float(total_return - buy_hold_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown_pct': float(max_drawdown),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'trade_win_rate_pct': float(trade_win_rate),
            'capital_history': [float(c) for c in capital_history]
        }


def load_model(checkpoint_path, args, device='cuda'):
    """Load trained model from checkpoint"""
    # Load content
    args.content = load_content(args)
    
    # Initialize model
    model = TimeLLM.Model(args).float()
    
    # Load checkpoint
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint')
    if os.path.exists(checkpoint_file):
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        print(f"Loaded model from {checkpoint_file}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    model = model.to(device)
    model.eval()
    
    return model


def run_evaluation(args, checkpoint_path, device='cuda'):
    """Run full evaluation pipeline"""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Load model
    model = load_model(checkpoint_path, args, device)
    
    # Load data
    train_data, train_loader = data_provider(args, 'train')
    test_data, test_loader = data_provider(args, 'test')
    
    # Initialize evaluator
    evaluator = StockEvaluator(model, args, device)
    
    results = {}
    
    # Evaluate on training data
    print("\n--- Evaluating on Training Data ---")
    train_preds, train_actuals = evaluator.predict(train_loader, train_data)
    results['train'] = {
        'metrics': evaluator.calculate_metrics(train_preds, train_actuals),
        'directional': evaluator.calculate_directional_accuracy(train_preds, train_actuals),
        'trading': evaluator.calculate_trading_pnl(train_preds, train_actuals)
    }
    
    # Evaluate on test data
    print("\n--- Evaluating on Test Data ---")
    test_preds, test_actuals = evaluator.predict(test_loader, test_data)
    results['test'] = {
        'metrics': evaluator.calculate_metrics(test_preds, test_actuals),
        'directional': evaluator.calculate_directional_accuracy(test_preds, test_actuals),
        'trading': evaluator.calculate_trading_pnl(test_preds, test_actuals)
    }
    
    return results, train_preds, train_actuals, test_preds, test_actuals


def print_results(results):
    """Print evaluation results in formatted way"""
    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f" {split.upper()} SET RESULTS")
        print(f"{'='*60}")
        
        # Metrics
        print("\n📊 Prediction Metrics:")
        metrics = results[split]['metrics']
        print(f"   MSE:  {metrics['MSE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE:  {metrics['MAE']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   R²:   {metrics['R2']:.4f}")
        
        # Directional accuracy
        print("\n🎯 Directional Accuracy (Win Rate):")
        dir_acc = results[split]['directional']
        print(f"   Overall Accuracy: {dir_acc['directional_accuracy']:.2f}%")
        print(f"   Correct/Total: {dir_acc['correct_predictions']}/{dir_acc['total_predictions']}")
        print(f"   Up Precision:   {dir_acc['up_precision']:.2f}%")
        print(f"   Down Precision: {dir_acc['down_precision']:.2f}%")
        
        # Trading P&L
        print("\n💰 Trading Performance:")
        trading = results[split]['trading']
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
    
    # Model config
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--prediction_type', type=str, default='short_term',
                        choices=['short_term', 'mid_term'],
                        help='Prediction type')
    
    # Data config
    parser.add_argument('--data', type=str, default='Stock', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/stock/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='vcb_stock_indicators.csv',
                        help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task')
    parser.add_argument('--target', type=str, default='Adj Close', help='target feature')
    parser.add_argument('--freq', type=str, default='d', help='time frequency')
    
    # Model parameters
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=30, help='label length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction length')
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=3, help='attention factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=4, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=1, help='domain prompts')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default=768, help='LLM dimension')
    parser.add_argument('--llm_layers', type=int, default=6, help='LLM layers')
    
    # Other
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    parser.add_argument('--output_attention', action='store_true', help='output attention')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='seasonal patterns')
    parser.add_argument('--percent', type=int, default=100, help='data percentage')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set prediction length based on type
    if args.prediction_type == 'short_term':
        args.pred_len = 1
    else:
        args.pred_len = 60
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run evaluation
    results, train_preds, train_actuals, test_preds, test_actuals = run_evaluation(
        args, args.checkpoint_path, device
    )
    
    # Print results
    print_results(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    results_file = os.path.join(args.output_dir, f'evaluation_{args.prediction_type}_{timestamp}.json')
    
    # Remove capital_history for cleaner JSON
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
    print(f"Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

