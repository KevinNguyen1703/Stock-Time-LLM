"""
Stock Data Preparation V0 - Baseline
Creates basic dataset with original indicators only (no momentum features)
For comparison with enhanced V2 version
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json


def parse_date(date_str):
    """Parse date string to datetime"""
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
    return bb_position.clip(0, 1)


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


def prepare_stock_data_v0(input_file, output_dir):
    """
    Prepare V0 baseline stock data from raw CSV
    Uses only basic indicators without momentum features
    """
    # Read raw data
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Columns: {list(df.columns)}")
    
    # Parse dates
    if 'Date' in df.columns:
        df['date'] = df['Date'].apply(parse_date)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("No date column found!")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Determine price column
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        raise ValueError("No price column found!")
    
    print(f"\nCalculating basic indicators using {price_col}...")
    
    # Calculate basic indicators
    df['RSI'] = calculate_rsi(df[price_col], period=14)
    df['MACD'] = calculate_macd(df[price_col])
    df['BB_Position'] = calculate_bollinger_position(df[price_col])
    df['ROC'] = calculate_roc(df[price_col], period=10)
    
    # Volume normalization
    if 'Volume' in df.columns:
        df['Volume_Norm'] = normalize_volume(df['Volume'])
    else:
        df['Volume_Norm'] = 0
    
    # Rename target column to standard name
    df['Adj Close'] = df[price_col]
    
    # Drop NaN values
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"After cleaning: {len(df_clean)} rows")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select V0 features (basic indicators only)
    features_v0 = ['date', 'RSI', 'MACD', 'BB_Position', 'Volume_Norm', 'ROC', 'Adj Close']
    
    df_v0 = df_clean[features_v0].copy()
    
    # Save V0 dataset
    output_file = os.path.join(output_dir, 'vcb_stock_indicators_v0.csv')
    df_v0.to_csv(output_file, index=False)
    print(f"\nSaved V0 dataset to: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("V0 Baseline Dataset Statistics:")
    print("="*60)
    print(f"Total samples: {len(df_v0)}")
    print(f"Date range: {df_v0['date'].min()} to {df_v0['date'].max()}")
    print(f"Features: 5 (RSI, MACD, BB_Position, Volume_Norm, ROC)")
    print(f"Target: Adj Close")
    
    numeric_cols = df_v0.select_dtypes(include=[np.number]).columns
    print(f"\nFeature Statistics:")
    print(df_v0[numeric_cols].describe().round(2))
    
    # Save metadata
    metadata = {
        'version': 'v0_baseline',
        'features': ['RSI', 'MACD', 'BB_Position', 'Volume_Norm', 'ROC'],
        'target': 'Adj Close',
        'total_samples': len(df_v0),
        'date_range': f"{df_v0['date'].min()} to {df_v0['date'].max()}",
        'description': 'Baseline dataset with basic indicators only (no momentum features)'
    }
    
    with open(os.path.join(output_dir, 'dataset_metadata_v0.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return df_v0


def generate_basic_prompts(df, output_dir, seq_len=60, pred_len=1):
    """Generate basic prompts (simple statistical description, no trading signals)"""
    
    prompts = {}
    total_windows = len(df) - seq_len - pred_len + 1
    
    print(f"\nGenerating basic prompts for {total_windows} windows...")
    
    for i in range(total_windows):
        window = df.iloc[i:i+seq_len]
        
        rsi = window['RSI'].iloc[-1]
        macd = window['MACD'].iloc[-1]
        bb_pos = window['BB_Position'].iloc[-1]
        roc = window['ROC'].iloc[-1]
        
        price_start = window['Adj Close'].iloc[0]
        price_end = window['Adj Close'].iloc[-1]
        price_change = (price_end - price_start) / price_start * 100
        
        # Simple descriptive prompt (no trading signals)
        prompt = (
            f"VCB stock analysis for the past 60 days: "
            f"RSI is {rsi:.1f}, MACD is {macd:.2f}, "
            f"Bollinger position is {bb_pos:.2f}, ROC is {roc:.2f}%. "
            f"Price changed {price_change:+.2f}% during this period."
        )
        
        prompts[str(i)] = prompt
    
    # Save prompts
    output_file = os.path.join(output_dir, 'prompts_v0_short_term.json')
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to: {output_file}")
    
    # Mid-term prompts
    prompts_mid = {}
    total_windows_mid = len(df) - seq_len - 60 + 1
    for i in range(total_windows_mid):
        if str(i) in prompts:
            prompts_mid[str(i)] = prompts[str(i)]
    
    output_file_mid = os.path.join(output_dir, 'prompts_v0_mid_term.json')
    with open(output_file_mid, 'w') as f:
        json.dump(prompts_mid, f, indent=2)
    print(f"Saved {len(prompts_mid)} mid-term prompts to: {output_file_mid}")
    
    return prompts


if __name__ == "__main__":
    # Use the raw VCB data file
    input_file = "./gia_VCB_2020-2025.csv"
    output_dir = "./dataset/dataset/stock"
    
    # Prepare V0 data
    df_v0 = prepare_stock_data_v0(input_file, output_dir)
    
    if df_v0 is not None:
        # Generate basic prompts
        generate_basic_prompts(df_v0, output_dir)
        
        print("\n" + "="*60)
        print("V0 Baseline Data preparation completed!")
        print("="*60)
        print("\nFiles created:")
        print("  - vcb_stock_indicators_v0.csv (basic indicators)")
        print("  - prompts_v0_short_term.json (basic prompts)")
        print("  - prompts_v0_mid_term.json (basic prompts)")
        print("  - dataset_metadata_v0.json (metadata)")

