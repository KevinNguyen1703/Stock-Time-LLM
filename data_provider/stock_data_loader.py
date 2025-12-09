"""
Custom Stock Data Loader for Time-LLM with Dynamic Prompts
Supports dynamic per-sample prompts (professor advice) for stock price prediction
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import json

warnings.filterwarnings('ignore')


class Dataset_Stock(Dataset):
    """
    Stock dataset with technical indicators for Time-LLM
    Supports both short-term (1-day) and mid-term (60-day) predictions
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='vcb_stock_indicators.csv',
                 target='Adj Close', scale=True, timeenc=0, freq='d', percent=100,
                 seasonal_patterns=None, prompt_data_path=None):
        """
        Args:
            root_path: Root path for data
            flag: 'train', 'val', or 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M' for multivariate, 'S' for univariate, 'MS' for multivariate input with univariate output
            data_path: Path to stock data CSV
            target: Target column name ('Adj Close')
            scale: Whether to scale data
            timeenc: Time encoding type
            freq: Frequency ('d' for daily)
            percent: Percentage of training data to use
            prompt_data_path: Path to prompt data (JSON with market analysis)
        """
        if size is None:
            self.seq_len = 60
            self.label_len = 30
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.prompt_data_path = prompt_data_path
        
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Expected columns: date, RSI, MACD, BB_Position, Volume_Norm, ROC, Adj Close
        cols = list(df_raw.columns)
        
        # Ensure target is last column
        if self.target in cols:
            cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        
        # Reorder: date, features, target
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # Train/val/test split (70/10/20)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        
        # Select features based on mode
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # All columns except date
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        # Scale data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Process time stamps
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = 0  # Daily data, no hourly component
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        # Store raw dates for prompt generation
        self.dates = df_raw['date'][border1:border2].values
        
        # Store raw unscaled data for analysis
        self.raw_data = df_data[border1:border2].values
        
        # Store border info for prompt index mapping
        self.border1 = border1
        self.border2 = border2
        
        # Load prompt data if available
        self.prompts = None
        if self.prompt_data_path:
            prompt_file = os.path.join(self.root_path, self.prompt_data_path)
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    self.prompts = json.load(f)
                print(f"Loaded {len(self.prompts)} prompts from {prompt_file}")
    
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_prompt(self, index):
        """Get the dynamic prompt for a specific sample index"""
        s_begin = index % self.tot_len
        prompt_key = str(s_begin)
        
        if self.prompts is not None and prompt_key in self.prompts:
            return self.prompts[prompt_key]
        
        # Generate statistical prompt as fallback
        return self._generate_statistical_prompt(s_begin)
    
    def _generate_statistical_prompt(self, s_begin):
        """Generate a statistical prompt based on window data"""
        s_end = s_begin + self.seq_len
        window_raw = self.raw_data[s_begin:s_end]
        
        # Feature indices: RSI=0, MACD=1, BB_Position=2, Volume_Norm=3, ROC=4, Adj Close=5
        rsi = window_raw[:, 0]
        macd = window_raw[:, 1]
        bb_pos = window_raw[:, 2]
        volume = window_raw[:, 3]
        roc = window_raw[:, 4]
        price = window_raw[:, 5]
        
        # Generate analysis
        rsi_current = rsi[-1]
        rsi_signal = "overbought" if rsi_current > 70 else ("oversold" if rsi_current < 30 else "neutral")
        
        macd_current = macd[-1]
        macd_signal = "bullish momentum" if macd_current > 0 else "bearish momentum"
        
        bb_current = bb_pos[-1]
        bb_signal = "near upper band (potentially overbought)" if bb_current > 0.8 else (
            "near lower band (potentially oversold)" if bb_current < 0.2 else "within normal range")
        
        price_change = ((price[-1] - price[0]) / price[0]) * 100
        trend = "upward" if price_change > 0 else "downward"
        
        volume_current = volume[-1]
        volume_signal = "high trading activity" if volume_current > 1 else (
            "low trading activity" if volume_current < -1 else "normal trading activity")
        
        prompt = (
            f"VCB Stock Analysis for the past 60 trading days: "
            f"The stock shows a {trend} trend with {abs(price_change):.2f}% price change. "
            f"RSI at {rsi_current:.2f} indicates {rsi_signal} conditions. "
            f"MACD shows {macd_signal} (current: {macd_current:.4f}). "
            f"Bollinger Band position at {bb_current:.2f} suggests the price is {bb_signal}. "
            f"Volume analysis indicates {volume_signal}. "
            f"Rate of Change (ROC) is {roc[-1]:.2f}%, suggesting "
            f"{'positive' if roc[-1] > 0 else 'negative'} momentum. "
            f"Current closing price: {price[-1]:.2f} VND."
        )
        
        return prompt
    
    def get_window_data(self, index):
        """Get raw window data for prompt generation"""
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        
        window_data = {
            'raw_data': self.raw_data[s_begin:s_end],
            'dates': self.dates[s_begin:s_end],
            'start_date': self.dates[s_begin],
            'end_date': self.dates[s_end - 1]
        }
        return window_data


class Dataset_Stock_WithPrompt(Dataset_Stock):
    """
    Extended Stock Dataset that returns dynamic prompts with each sample
    Used for training Time-LLM with per-sample professor advice
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_names = ['RSI', 'MACD', 'BB_Position', 'Volume_Norm', 'ROC', 'Adj Close']
    
    def __getitem__(self, index):
        """
        Returns:
            seq_x: Input sequence
            seq_y: Target sequence
            seq_x_mark: Input time encoding
            seq_y_mark: Target time encoding
            prompt: Dynamic prompt string for this sample
        """
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # Get dynamic prompt
        prompt = self.get_prompt(index)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt


def stock_collate_fn(batch):
    """
    Custom collate function for Dataset_Stock_WithPrompt
    Handles the string prompts properly
    """
    import torch
    
    seq_x = torch.stack([torch.tensor(item[0]) for item in batch])
    seq_y = torch.stack([torch.tensor(item[1]) for item in batch])
    seq_x_mark = torch.stack([torch.tensor(item[2]) for item in batch])
    seq_y_mark = torch.stack([torch.tensor(item[3]) for item in batch])
    prompts = [item[4] for item in batch]  # Keep as list of strings
    
    return seq_x, seq_y, seq_x_mark, seq_y_mark, prompts


def get_stock_content(data_path, seq_len=60, pred_len=1):
    """
    Generate dataset description for Time-LLM prompt
    """
    content = (
        f"This dataset contains VCB (Vietcombank) stock price data with technical indicators. "
        f"Features include: RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), "
        f"BB_Position (Bollinger Band Position), Volume (normalized trading volume), "
        f"ROC (Rate of Change), and Adj Close (Adjusted Closing Price). "
        f"The task is to predict the stock's adjusted closing price for the next "
        f"{'day' if pred_len == 1 else f'{pred_len} days'} based on the past {seq_len} days of data. "
        f"RSI above 70 typically indicates overbought conditions, while below 30 indicates oversold. "
        f"Positive MACD suggests bullish momentum, negative suggests bearish. "
        f"BB_Position near 1 suggests price is near upper band, near 0 suggests lower band."
    )
    return content
