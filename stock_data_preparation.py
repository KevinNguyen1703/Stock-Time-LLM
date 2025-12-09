"""
Stock Data Preparation Script
Calculates technical indicators: RSI, MACD, BB_Position, Volume, ROC
Prepares datasets for short-term (1-day) and mid-term (60-day) predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


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
    """Calculate MACD and return MACD line (difference between MACD and signal)"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - signal_line
    return macd_histogram


def calculate_bollinger_position(prices, period=20, num_std=2):
    """
    Calculate Bollinger Band Position
    Returns value between 0 and 1 indicating position within bands
    0 = at lower band, 1 = at upper band, 0.5 = at middle
    """
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


def prepare_stock_data(input_file, output_dir):
    """
    Prepare stock data with technical indicators for Time-LLM training
    
    Args:
        input_file: Path to raw stock CSV file
        output_dir: Directory to save processed datasets
    """
    # Read raw data
    df = pd.read_csv(input_file)
    
    # Parse dates
    df['Date'] = df['Date'].apply(parse_date)
    
    # Sort by date (oldest first for time series)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    
    # RSI (14-period)
    df['RSI'] = calculate_rsi(df['Adj Close'], period=14)
    
    # MACD
    df['MACD'] = calculate_macd(df['Adj Close'])
    
    # Bollinger Band Position
    df['BB_Position'] = calculate_bollinger_position(df['Adj Close'])
    
    # Rate of Change (10-period)
    df['ROC'] = calculate_roc(df['Adj Close'], period=10)
    
    # Normalized Volume
    df['Volume_Norm'] = normalize_volume(df['Volume'])
    
    # Drop NaN values (from indicator calculations)
    df = df.dropna().reset_index(drop=True)
    
    # Rename date column for Time-LLM compatibility
    df = df.rename(columns={'Date': 'date'})
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dataset with selected features
    # Features: RSI, MACD, BB_Position, Volume_Norm, ROC, Adj Close (target)
    features = ['date', 'RSI', 'MACD', 'BB_Position', 'Volume_Norm', 'ROC', 'Adj Close']
    df_processed = df[features].copy()
    
    # Save main processed dataset
    main_output = os.path.join(output_dir, 'vcb_stock_indicators.csv')
    df_processed.to_csv(main_output, index=False)
    print(f"Saved main dataset to: {main_output}")
    
    # Create short-term prediction dataset (predict next 1 day)
    short_term_file = os.path.join(output_dir, 'vcb_stock_short_term.csv')
    df_processed.to_csv(short_term_file, index=False)
    print(f"Saved short-term dataset to: {short_term_file}")
    
    # Create mid-term prediction dataset (predict next 60 days)
    mid_term_file = os.path.join(output_dir, 'vcb_stock_mid_term.csv')
    df_processed.to_csv(mid_term_file, index=False)
    print(f"Saved mid-term dataset to: {mid_term_file}")
    
    # Print dataset statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    print(f"Total samples: {len(df_processed)}")
    print(f"Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
    print(f"\nFeature Statistics:")
    print(df_processed.describe())
    
    # Save metadata
    metadata = {
        'features': ['RSI', 'MACD', 'BB_Position', 'Volume_Norm', 'ROC', 'Adj Close'],
        'target': 'Adj Close',
        'total_samples': len(df_processed),
        'date_range': f"{df_processed['date'].min()} to {df_processed['date'].max()}",
        'short_term_config': {
            'seq_len': 60,
            'pred_len': 1,
            'description': 'Use last 60 days to predict next day'
        },
        'mid_term_config': {
            'seq_len': 60,
            'pred_len': 60,
            'description': 'Use last 60 days to predict next 60 days'
        }
    }
    
    import json
    metadata_file = os.path.join(output_dir, 'dataset_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nSaved metadata to: {metadata_file}")
    
    return df_processed


def create_sample_analysis_data(df, output_dir):
    """
    Create analysis data for ChatGPT prompt generation
    Splits data into windows and calculates statistics for each window
    """
    window_size = 60
    analysis_data = []
    
    for i in range(window_size, len(df)):
        window = df.iloc[i-window_size:i]
        
        # Calculate window statistics
        stats = {
            'window_end_date': str(df.iloc[i]['date']),
            'window_start_date': str(df.iloc[i-window_size]['date']),
            'rsi_mean': window['RSI'].mean(),
            'rsi_current': window['RSI'].iloc[-1],
            'rsi_trend': 'overbought' if window['RSI'].iloc[-1] > 70 else ('oversold' if window['RSI'].iloc[-1] < 30 else 'neutral'),
            'macd_mean': window['MACD'].mean(),
            'macd_current': window['MACD'].iloc[-1],
            'macd_signal': 'bullish' if window['MACD'].iloc[-1] > 0 else 'bearish',
            'bb_position_mean': window['BB_Position'].mean(),
            'bb_position_current': window['BB_Position'].iloc[-1],
            'volume_trend': 'high' if window['Volume_Norm'].iloc[-1] > 1 else ('low' if window['Volume_Norm'].iloc[-1] < -1 else 'normal'),
            'roc_current': window['ROC'].iloc[-1],
            'price_change_pct': ((window['Adj Close'].iloc[-1] - window['Adj Close'].iloc[0]) / window['Adj Close'].iloc[0]) * 100,
            'price_volatility': window['Adj Close'].std() / window['Adj Close'].mean() * 100,
            'current_price': window['Adj Close'].iloc[-1],
            'price_min': window['Adj Close'].min(),
            'price_max': window['Adj Close'].max()
        }
        analysis_data.append(stats)
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_file = os.path.join(output_dir, 'window_analysis.csv')
    analysis_df.to_csv(analysis_file, index=False)
    print(f"Saved window analysis to: {analysis_file}")
    
    return analysis_df


if __name__ == "__main__":
    # Input and output paths
    input_file = "./gia_VCB_2020-2025.csv"
    output_dir = "./dataset/dataset/stock"
    
    # Prepare the data
    df = prepare_stock_data(input_file, output_dir)
    
    # Create analysis data for prompt generation
    create_sample_analysis_data(df, output_dir)
    
    print("\n" + "="*50)
    print("Data preparation completed!")
    print("="*50)

