"""
Dynamic Prompt Generator for Stock Market Analysis
Uses ChatGPT API to generate professor-like market analysis and trading advice
Based on technical indicators for each time window
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
import argparse

# Optional: Use OpenAI API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Using fallback prompt generation.")


class ProfessorPromptGenerator:
    """
    Generate professor-like market analysis prompts using ChatGPT API
    These prompts serve as expert guidance during Time-LLM training
    """
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize the prompt generator
        
        Args:
            api_key: OpenAI API key (can also be set via OPENAI_API_KEY env variable)
            model: ChatGPT model to use (gpt-3.5-turbo or gpt-4)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            print(f"OpenAI client initialized with model: {model}")
        else:
            print("Using fallback statistical prompt generation (professor-style)")
    
    def analyze_window(self, window_data, feature_names):
        """
        Analyze a window of technical indicator data
        
        Args:
            window_data: numpy array of shape (window_size, num_features)
            feature_names: list of feature names
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        for i, name in enumerate(feature_names):
            data = window_data[:, i]
            analysis[name] = {
                'current': float(data[-1]),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'trend': 'up' if data[-1] > data[0] else 'down',
                'change_pct': float((data[-1] - data[0]) / (abs(data[0]) + 1e-8) * 100),
                'recent_5_avg': float(np.mean(data[-5:])),
                'is_increasing': bool(np.mean(np.diff(data[-10:])) > 0)
            }
        
        return analysis
    
    def generate_chatgpt_prompt(self, analysis, start_date, end_date, pred_len=1):
        """
        Generate a professor-like analysis using ChatGPT API
        
        Args:
            analysis: Dictionary with technical indicator analysis
            start_date: Start date of the analysis window
            end_date: End date of the analysis window
            pred_len: Prediction length (1 for short-term, 60 for mid-term)
        
        Returns:
            Generated prompt string
        """
        if not self.client:
            return self.generate_professor_prompt(analysis, start_date, end_date, pred_len)
        
        # Create system message - Professor persona
        system_message = """You are Professor Nguyen, a renowned Vietnamese stock market expert with 30 years of experience in technical analysis. 
You specialize in VCB (Vietcombank) stock and Vietnamese banking sector analysis.

Your task is to provide brief, actionable market analysis and prediction guidance based on technical indicators.
Speak in a professorial but accessible tone. Be direct about what the indicators suggest.
Keep responses to 2-3 sentences max. Focus on:
1. What the key indicators tell us
2. What this suggests for the price direction
3. Any caution or confidence level in the prediction

Do NOT give specific price targets. Focus on directional guidance and confidence."""
        
        # Build indicator summary
        rsi = analysis['RSI']
        macd = analysis['MACD']
        bb = analysis['BB_Position']
        volume = analysis['Volume_Norm']
        roc = analysis['ROC']
        price = analysis['Adj Close']
        
        # Create user message with analysis data
        user_message = f"""Analyze VCB stock from {start_date} to {end_date}:

RSI: {rsi['current']:.1f} (trend: {rsi['trend']}, avg: {rsi['mean']:.1f})
MACD: {macd['current']:.4f} (trend: {macd['trend']})
BB Position: {bb['current']:.2f} (0=lower, 1=upper band)
Volume (Z-score): {volume['current']:.2f}
ROC: {roc['current']:.2f}%
Price Change: {price['change_pct']:.2f}% over period

Provide your professor's analysis for predicting the next {'day' if pred_len == 1 else f'{pred_len} days'}."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"ChatGPT API error: {e}")
            return self.generate_professor_prompt(analysis, start_date, end_date, pred_len)
    
    def generate_professor_prompt(self, analysis, start_date, end_date, pred_len=1):
        """
        Generate a professor-like prompt using statistical analysis (fallback)
        This creates expert-sounding advice based on technical indicators
        """
        rsi = analysis['RSI']
        macd = analysis['MACD']
        bb = analysis['BB_Position']
        volume = analysis['Volume_Norm']
        roc = analysis['ROC']
        price = analysis['Adj Close']
        
        # Determine market condition
        signals = []
        confidence_factors = []
        
        # RSI Analysis
        if rsi['current'] > 70:
            signals.append("bearish")
            confidence_factors.append(f"RSI at {rsi['current']:.1f} shows overbought conditions - expect potential pullback")
        elif rsi['current'] < 30:
            signals.append("bullish")
            confidence_factors.append(f"RSI at {rsi['current']:.1f} indicates oversold territory - watch for reversal")
        elif rsi['current'] > 50:
            if rsi['is_increasing']:
                signals.append("bullish")
                confidence_factors.append(f"RSI trending up at {rsi['current']:.1f} supports upward momentum")
            else:
                confidence_factors.append(f"RSI at {rsi['current']:.1f} showing mixed signals")
        else:
            if not rsi['is_increasing']:
                signals.append("bearish")
                confidence_factors.append(f"RSI declining below 50 suggests weakening momentum")
        
        # MACD Analysis
        if macd['current'] > 0 and macd['is_increasing']:
            signals.append("bullish")
            confidence_factors.append("MACD positive and rising confirms bullish momentum")
        elif macd['current'] < 0 and not macd['is_increasing']:
            signals.append("bearish")
            confidence_factors.append("MACD negative and falling reinforces bearish pressure")
        elif macd['current'] > 0 and not macd['is_increasing']:
            confidence_factors.append("MACD positive but losing strength - momentum may be fading")
        elif macd['current'] < 0 and macd['is_increasing']:
            signals.append("bullish")
            confidence_factors.append("MACD turning up from negative territory - potential trend reversal")
        
        # Bollinger Band Analysis
        if bb['current'] > 0.85:
            signals.append("bearish")
            confidence_factors.append(f"Price near upper Bollinger Band ({bb['current']:.2f}) suggests resistance ahead")
        elif bb['current'] < 0.15:
            signals.append("bullish")
            confidence_factors.append(f"Price near lower Bollinger Band ({bb['current']:.2f}) indicates support level")
        
        # Volume Analysis
        if volume['current'] > 1.5:
            confidence_factors.append("Elevated volume suggests strong conviction in current move")
        elif volume['current'] < -1.0:
            confidence_factors.append("Low volume indicates weak participation - trend may lack follow-through")
        
        # ROC Analysis
        if abs(roc['current']) > 5:
            if roc['current'] > 0:
                signals.append("bullish")
                confidence_factors.append(f"Strong ROC at {roc['current']:.1f}% shows significant upward momentum")
            else:
                signals.append("bearish")
                confidence_factors.append(f"Negative ROC at {roc['current']:.1f}% indicates selling pressure")
        
        # Determine overall sentiment
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")
        
        if bullish_count > bearish_count:
            sentiment = "bullish"
            direction = "upward"
            confidence = "moderate to high" if bullish_count >= 3 else "moderate"
        elif bearish_count > bullish_count:
            sentiment = "bearish"
            direction = "downward"
            confidence = "moderate to high" if bearish_count >= 3 else "moderate"
        else:
            sentiment = "neutral"
            direction = "sideways"
            confidence = "low"
        
        # Select top 2 most relevant factors
        key_factors = confidence_factors[:2] if len(confidence_factors) >= 2 else confidence_factors
        factors_text = ". ".join(key_factors) if key_factors else "Mixed signals present"
        
        # Generate professor-style advice
        time_horizon = "short-term" if pred_len == 1 else "medium-term"
        
        professor_advice = (
            f"Based on my analysis of VCB over the past 60 days, the technical picture is {sentiment}. "
            f"{factors_text}. "
            f"For the {time_horizon} ({pred_len} {'day' if pred_len == 1 else 'days'}), "
            f"I expect {direction} price movement with {confidence} confidence. "
        )
        
        # Add specific caution for extreme conditions
        if rsi['current'] > 80 or rsi['current'] < 20:
            professor_advice += "However, extreme RSI readings warrant caution - reversals can be sharp. "
        
        if abs(price['change_pct']) > 10:
            professor_advice += f"The {abs(price['change_pct']):.1f}% move over this period is significant - watch for consolidation. "
        
        return professor_advice
    
    def generate_prompts_for_dataset(self, data_path, output_path, seq_len=60, pred_len=1, 
                                     use_api=True, rate_limit_delay=0.5):
        """
        Generate prompts for all windows in the dataset
        
        Args:
            data_path: Path to the processed stock data CSV
            output_path: Path to save generated prompts JSON
            seq_len: Sequence length (window size)
            pred_len: Prediction length
            use_api: Whether to use ChatGPT API
            rate_limit_delay: Delay between API calls to avoid rate limiting
        """
        # Load data
        df = pd.read_csv(data_path)
        feature_names = ['RSI', 'MACD', 'BB_Position', 'Volume_Norm', 'ROC', 'Adj Close']
        
        # Extract feature data
        data = df[feature_names].values
        dates = df['date'].values
        
        prompts = {}
        total_windows = len(df) - seq_len - pred_len + 1
        
        print(f"Generating professor prompts for {total_windows} windows...")
        print(f"Using {'ChatGPT API' if (use_api and self.client) else 'statistical analysis'}...")
        
        for i in tqdm(range(total_windows)):
            window_data = data[i:i+seq_len]
            start_date = dates[i]
            end_date = dates[i+seq_len-1]
            
            # Analyze window
            analysis = self.analyze_window(window_data, feature_names)
            
            # Generate prompt
            if use_api and self.client:
                prompt = self.generate_chatgpt_prompt(analysis, start_date, end_date, pred_len)
                time.sleep(rate_limit_delay)  # Rate limiting
            else:
                prompt = self.generate_professor_prompt(analysis, start_date, end_date, pred_len)
            
            prompts[str(i)] = prompt
            
            # Save periodically
            if (i + 1) % 100 == 0:
                with open(output_path, 'w') as f:
                    json.dump(prompts, f, indent=2)
        
        # Final save
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        print(f"Saved {len(prompts)} professor prompts to {output_path}")
        return prompts


# Keep backward compatibility
PromptGenerator = ProfessorPromptGenerator


def create_stock_prompt_file(output_dir):
    """Create a prompt description file for the stock dataset (for Time-LLM)"""
    content = """VCB (Vietcombank) Stock Price Prediction Dataset

This dataset contains daily stock price data with technical indicators for VCB, one of Vietnam's largest commercial banks.

Technical Indicators:
- RSI (Relative Strength Index): Momentum oscillator measuring speed and magnitude of price changes. Values > 70 indicate overbought, < 30 indicate oversold.
- MACD (Moving Average Convergence Divergence): Trend-following momentum indicator. Positive values suggest bullish momentum, negative values suggest bearish.
- BB_Position (Bollinger Band Position): Shows where price is relative to Bollinger Bands. 0 = at lower band, 1 = at upper band, 0.5 = at middle band.
- Volume (Normalized): Z-score normalized trading volume. High values indicate unusual trading activity.
- ROC (Rate of Change): Percentage change in price over a period, measuring momentum.
- Adj Close: Adjusted closing price accounting for dividends and splits.

Market Context:
- VCB is a blue-chip stock in the Vietnamese market (HOSE)
- Banking sector in Vietnam shows sensitivity to monetary policy and economic growth
- Stock exhibits typical trading patterns with support and resistance levels

The model should use these indicators to predict future adjusted closing prices."""

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'Stock.txt'), 'w') as f:
        f.write(content)
    print(f"Created Stock.txt prompt file in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate professor-style prompts for stock prediction')
    parser.add_argument('--data_path', type=str, default='./dataset/dataset/stock/vcb_stock_indicators.csv',
                        help='Path to stock data CSV')
    parser.add_argument('--output_dir', type=str, default='./dataset/dataset/stock',
                        help='Output directory for prompts')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length')
    parser.add_argument('--use_api', action='store_true', help='Use ChatGPT API')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='ChatGPT model')
    
    args = parser.parse_args()
    
    # Create prompt file for Time-LLM
    create_stock_prompt_file('./dataset/prompt_bank')
    
    # Generate prompts
    generator = ProfessorPromptGenerator(api_key=args.api_key, model=args.model)
    
    # Generate for short-term prediction (pred_len=1)
    short_term_output = os.path.join(args.output_dir, 'prompts_short_term.json')
    generator.generate_prompts_for_dataset(
        args.data_path, 
        short_term_output,
        seq_len=args.seq_len,
        pred_len=1,
        use_api=args.use_api
    )
    
    # Generate for mid-term prediction (pred_len=60)
    mid_term_output = os.path.join(args.output_dir, 'prompts_mid_term.json')
    generator.generate_prompts_for_dataset(
        args.data_path,
        mid_term_output,
        seq_len=args.seq_len,
        pred_len=60,
        use_api=args.use_api
    )


if __name__ == "__main__":
    main()
