"""
Stock Data Preparation V2 - Improved for Better Directional Prediction
Adds momentum features and price change targets to help model learn direction
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json


def parse_date(date_str):
    """Parse date string in format 'Mon DD YYYY' to datetime"""
    try:
        return pd.to_datetime(date_str, format='%b %d %Y')
    except:
        return pd.to_datetime(date_str)


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - signal_line
    return macd_histogram


def calculate_bollinger_position(prices, period=20, num_std=2):
    """Calculate Bollinger Band Position (0-1 scale)"""
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position


def calculate_roc(prices, period=10):
    """Calculate Rate of Change"""
    roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    return roc


def normalize_volume(volume):
    """Normalize volume using rolling z-score"""
    rolling_mean = volume.rolling(window=20).mean()
    rolling_std = volume.rolling(window=20).std()
    normalized = (volume - rolling_mean) / rolling_std
    return normalized


# ============== NEW MOMENTUM FEATURES ==============

def calculate_price_momentum(prices, periods=[1, 3, 5, 10]):
    """Calculate price momentum (returns) over multiple periods"""
    momentum = {}
    for p in periods:
        momentum[f'momentum_{p}d'] = prices.pct_change(p) * 100  # Percentage return
    return momentum


def calculate_ma_crossover(prices, short=5, long=20):
    """Calculate moving average crossover signal"""
    ma_short = prices.rolling(window=short).mean()
    ma_long = prices.rolling(window=long).mean()
    # Crossover signal: positive = bullish, negative = bearish
    crossover = (ma_short - ma_long) / ma_long * 100
    return crossover


def calculate_trend_strength(prices, period=14):
    """Calculate ADX-like trend strength (0-100)"""
    high = prices.rolling(period).max()
    low = prices.rolling(period).min()
    mid = (high + low) / 2
    
    # Trend strength based on position relative to range
    strength = abs(prices - mid) / (high - low + 1e-8) * 100
    return strength


def calculate_price_position(prices, period=20):
    """Calculate price position relative to recent range (0-1)"""
    high = prices.rolling(period).max()
    low = prices.rolling(period).min()
    position = (prices - low) / (high - low + 1e-8)
    return position


def calculate_volatility(prices, period=10):
    """Calculate realized volatility"""
    returns = prices.pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100  # Annualized
    return volatility


def calculate_direction_labels(prices, threshold=0.0):
    """Calculate next-day direction labels (1=up, 0=down, 0.5=flat)"""
    future_return = prices.shift(-1) / prices - 1
    direction = (future_return > threshold).astype(float)
    return direction, future_return * 100


def prepare_stock_data_v2(input_file, output_dir):
    """
    Prepare stock data with improved features for directional prediction
    """
    # Read raw data
    df = pd.read_csv(input_file)
    df['Date'] = df['Date'].apply(parse_date)
    df = df.sort_values('Date').reset_index(drop=True)
    
    print("Calculating technical indicators...")
    
    # Original indicators
    df['RSI'] = calculate_rsi(df['Adj Close'], period=14)
    df['MACD'] = calculate_macd(df['Adj Close'])
    df['BB_Position'] = calculate_bollinger_position(df['Adj Close'])
    df['ROC'] = calculate_roc(df['Adj Close'], period=10)
    df['Volume_Norm'] = normalize_volume(df['Volume'])
    
    # NEW: Momentum features
    print("Calculating momentum features...")
    momentum = calculate_price_momentum(df['Adj Close'], periods=[1, 3, 5, 10])
    for name, values in momentum.items():
        df[name] = values
    
    # NEW: Trend features
    df['MA_Crossover'] = calculate_ma_crossover(df['Adj Close'], short=5, long=20)
    df['Trend_Strength'] = calculate_trend_strength(df['Adj Close'], period=14)
    df['Price_Position'] = calculate_price_position(df['Adj Close'], period=20)
    df['Volatility'] = calculate_volatility(df['Adj Close'], period=10)
    
    # NEW: Direction labels for auxiliary training
    df['Direction'], df['Future_Return'] = calculate_direction_labels(df['Adj Close'])
    
    # Drop NaN values
    df = df.dropna().reset_index(drop=True)
    
    # Remove the last row (no future return available)
    df = df[:-1].reset_index(drop=True)
    
    # Rename date column
    df = df.rename(columns={'Date': 'date'})
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select features for training
    # Original features + new momentum features
    features_v2 = [
        'date',
        'RSI', 'MACD', 'BB_Position', 'Volume_Norm', 'ROC',  # Original
        'momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_10d',  # Momentum
        'MA_Crossover', 'Trend_Strength', 'Price_Position', 'Volatility',  # Trend
        'Adj Close'  # Target
    ]
    
    df_v2 = df[features_v2].copy()
    
    # Save V2 dataset
    output_file = os.path.join(output_dir, 'vcb_stock_indicators_v2.csv')
    df_v2.to_csv(output_file, index=False)
    print(f"Saved V2 dataset to: {output_file}")
    
    # Also save direction labels separately for auxiliary training
    direction_df = df[['date', 'Direction', 'Future_Return', 'Adj Close']].copy()
    direction_file = os.path.join(output_dir, 'vcb_direction_labels.csv')
    direction_df.to_csv(direction_file, index=False)
    print(f"Saved direction labels to: {direction_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("V2 Dataset Statistics:")
    print("="*60)
    print(f"Total samples: {len(df_v2)}")
    print(f"Date range: {df_v2['date'].min()} to {df_v2['date'].max()}")
    print(f"Features: {len(features_v2) - 2} (excluding date and target)")
    print(f"\nFeature Statistics:")
    print(df_v2.describe())
    
    # Calculate directional statistics
    up_days = (df['Direction'] == 1).sum()
    down_days = (df['Direction'] == 0).sum()
    print(f"\nDirection Statistics:")
    print(f"  Up days: {up_days} ({up_days/len(df)*100:.1f}%)")
    print(f"  Down days: {down_days} ({down_days/len(df)*100:.1f}%)")
    
    # Save metadata
    metadata = {
        'version': 'v2',
        'features': features_v2[1:-1],  # Exclude date and target
        'target': 'Adj Close',
        'total_samples': len(df_v2),
        'up_ratio': float(up_days / len(df)),
        'down_ratio': float(down_days / len(df)),
        'description': 'Improved dataset with momentum and trend features for better directional prediction'
    }
    
    with open(os.path.join(output_dir, 'dataset_metadata_v2.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return df_v2, df


def generate_improved_prompts(df, output_dir, seq_len=60, pred_len=1):
    """Generate improved prompts with more actionable trading signals"""
    
    prompts = {}
    total_windows = len(df) - seq_len - pred_len + 1
    
    print(f"\nGenerating improved prompts for {total_windows} windows...")
    
    for i in range(total_windows):
        window = df.iloc[i:i+seq_len]
        
        # Get current values
        rsi = window['RSI'].iloc[-1]
        macd = window['MACD'].iloc[-1]
        bb_pos = window['BB_Position'].iloc[-1]
        mom_1d = window['momentum_1d'].iloc[-1]
        mom_5d = window['momentum_5d'].iloc[-1]
        ma_cross = window['MA_Crossover'].iloc[-1]
        trend_str = window['Trend_Strength'].iloc[-1]
        volatility = window['Volatility'].iloc[-1]
        
        # Calculate trend over window
        price_start = window['Adj Close'].iloc[0]
        price_end = window['Adj Close'].iloc[-1]
        window_return = (price_end - price_start) / price_start * 100
        
        # Count bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0
        signal_details = []
        
        # RSI signals
        if rsi < 30:
            bullish_signals += 2  # Strong oversold
            signal_details.append("RSI oversold (strong buy signal)")
        elif rsi < 40:
            bullish_signals += 1
            signal_details.append("RSI approaching oversold")
        elif rsi > 70:
            bearish_signals += 2  # Strong overbought
            signal_details.append("RSI overbought (strong sell signal)")
        elif rsi > 60:
            bearish_signals += 1
            signal_details.append("RSI approaching overbought")
        
        # MACD signals
        if macd > 0 and mom_1d > 0:
            bullish_signals += 1
            signal_details.append("MACD positive with upward momentum")
        elif macd < 0 and mom_1d < 0:
            bearish_signals += 1
            signal_details.append("MACD negative with downward momentum")
        
        # MA Crossover
        if ma_cross > 2:
            bullish_signals += 1
            signal_details.append("Short MA above long MA (bullish crossover)")
        elif ma_cross < -2:
            bearish_signals += 1
            signal_details.append("Short MA below long MA (bearish crossover)")
        
        # Bollinger Band position
        if bb_pos < 0.2:
            bullish_signals += 1
            signal_details.append("Price near lower Bollinger Band (potential bounce)")
        elif bb_pos > 0.8:
            bearish_signals += 1
            signal_details.append("Price near upper Bollinger Band (potential resistance)")
        
        # Momentum confirmation
        if mom_1d > 1 and mom_5d > 2:
            bullish_signals += 1
            signal_details.append("Strong upward momentum on multiple timeframes")
        elif mom_1d < -1 and mom_5d < -2:
            bearish_signals += 1
            signal_details.append("Strong downward momentum on multiple timeframes")
        
        # Determine overall signal
        signal_diff = bullish_signals - bearish_signals
        
        if signal_diff >= 3:
            outlook = "STRONGLY BULLISH"
            confidence = "high"
            direction = "upward"
        elif signal_diff >= 1:
            outlook = "MODERATELY BULLISH"
            confidence = "moderate"
            direction = "upward"
        elif signal_diff <= -3:
            outlook = "STRONGLY BEARISH"
            confidence = "high"
            direction = "downward"
        elif signal_diff <= -1:
            outlook = "MODERATELY BEARISH"
            confidence = "moderate"
            direction = "downward"
        else:
            outlook = "NEUTRAL"
            confidence = "low"
            direction = "sideways"
        
        # Add volatility warning
        vol_warning = ""
        if volatility > 30:
            vol_warning = " High volatility suggests larger price swings possible."
        
        # Select top 2 signal details
        top_signals = signal_details[:2] if signal_details else ["Mixed technical signals"]
        signals_text = ". ".join(top_signals)
        
        # Generate prompt
        prompt = (
            f"TRADING SIGNAL: {outlook}. "
            f"Technical Analysis: {signals_text}. "
            f"The stock has moved {window_return:+.1f}% over the past 60 days. "
            f"Current momentum (1-day): {mom_1d:+.2f}%, (5-day): {mom_5d:+.2f}%. "
            f"RSI: {rsi:.1f}, MACD: {macd:.4f}. "
            f"Prediction confidence: {confidence}. "
            f"Expected direction: {direction}.{vol_warning}"
        )
        
        prompts[str(i)] = prompt
    
    # Save prompts
    output_file = os.path.join(output_dir, 'prompts_v2_short_term.json')
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} improved prompts to: {output_file}")
    
    # Also generate mid-term prompts
    prompts_mid = {}
    total_windows_mid = len(df) - seq_len - 60 + 1
    
    for i in range(total_windows_mid):
        # Reuse short-term analysis but adjust wording
        if str(i) in prompts:
            prompt = prompts[str(i)].replace("1-day", "60-day horizon")
            prompts_mid[str(i)] = prompt
    
    output_file_mid = os.path.join(output_dir, 'prompts_v2_mid_term.json')
    with open(output_file_mid, 'w') as f:
        json.dump(prompts_mid, f, indent=2)
    print(f"Saved {len(prompts_mid)} mid-term prompts to: {output_file_mid}")
    
    return prompts


if __name__ == "__main__":
    input_file = "./gia_VCB_2020-2025.csv"
    output_dir = "./dataset/dataset/stock"
    
    # Prepare V2 data
    df_v2, df_full = prepare_stock_data_v2(input_file, output_dir)
    
    # Generate improved prompts
    generate_improved_prompts(df_full, output_dir)
    
    print("\n" + "="*60)
    print("V2 Data preparation completed!")
    print("="*60)
    print("\nNew files created:")
    print("  - vcb_stock_indicators_v2.csv (with momentum features)")
    print("  - vcb_direction_labels.csv (direction labels)")
    print("  - prompts_v2_short_term.json (improved prompts)")
    print("  - prompts_v2_mid_term.json (improved prompts)")

