"""
FFT-based Patch Length Analysis for VCB Stock Data
===================================================
This script analyzes the VCB stock time series to discover dominant 
periodic patterns using Fast Fourier Transform (FFT), then recommends
optimal patch lengths for multi-scale patching.

Author: Generated for Time-LLM Stock Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')


def load_and_preprocess_data(filepath):
    """Load VCB stock data and preprocess."""
    df = pd.read_csv(filepath)
    
    # Parse dates (format: "Oct 24 2025")
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d %Y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} trading days")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    return df


def detrend_data(data):
    """Remove linear trend from data to focus on cyclical patterns."""
    n = len(data)
    x = np.arange(n)
    
    # Fit linear trend
    coeffs = np.polyfit(x, data, 1)
    trend = np.polyval(coeffs, x)
    
    # Remove trend
    detrended = data - trend
    
    return detrended, trend


def compute_fft_spectrum(data, sampling_rate=1):
    """
    Compute FFT and return frequencies, magnitudes, and periods.
    
    Args:
        data: Time series data (1D array)
        sampling_rate: Samples per unit time (1 = daily data)
    
    Returns:
        frequencies, magnitudes, periods
    """
    n = len(data)
    
    # Compute FFT
    fft_result = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(n, d=1/sampling_rate)
    magnitudes = np.abs(fft_result)
    
    # Normalize magnitudes
    magnitudes = magnitudes / n
    
    # Convert to periods (avoiding division by zero)
    with np.errstate(divide='ignore'):
        periods = np.where(frequencies > 0, 1 / frequencies, np.inf)
    
    return frequencies, magnitudes, periods


def find_dominant_periods(frequencies, magnitudes, periods, 
                          min_period=3, max_period=None, num_peaks=10):
    """
    Find dominant periods in the frequency spectrum.
    
    Args:
        frequencies: FFT frequencies
        magnitudes: FFT magnitudes
        periods: Corresponding periods
        min_period: Minimum period to consider
        max_period: Maximum period to consider
        num_peaks: Number of top peaks to return
    
    Returns:
        List of (period, magnitude) tuples sorted by magnitude
    """
    if max_period is None:
        max_period = len(periods) // 2
    
    # Filter valid periods
    valid_mask = (periods >= min_period) & (periods <= max_period) & np.isfinite(periods)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return []
    
    # Find peaks in the spectrum
    peak_indices, properties = signal.find_peaks(
        magnitudes[valid_indices], 
        height=np.percentile(magnitudes[valid_indices], 50),
        distance=2
    )
    
    if len(peak_indices) == 0:
        # Fall back to top magnitudes
        top_indices = np.argsort(magnitudes[valid_indices])[-num_peaks:]
        peak_indices = top_indices
    
    # Get actual indices in original arrays
    actual_indices = valid_indices[peak_indices]
    
    # Sort by magnitude
    sorted_idx = np.argsort(magnitudes[actual_indices])[::-1][:num_peaks]
    top_peaks = actual_indices[sorted_idx]
    
    results = [(int(round(periods[i])), magnitudes[i]) for i in top_peaks]
    
    return results


def recommend_patch_lengths(dominant_periods, seq_len=60, num_scales=3, 
                            min_patches_required=3):
    """
    Recommend patch lengths based on dominant periods.
    
    Args:
        dominant_periods: List of (period, magnitude) from FFT analysis
        seq_len: Model's input sequence length
        num_scales: Number of patch scales to use
        min_patches_required: Minimum patches needed per scale (default=3)
    
    The key constraint is: we need enough patches per scale for the model to learn.
    Formula: num_patches = (seq_len - patch_len) / stride + 2
    
    For min_patches_required=3 with 50% overlap (stride = patch_len/2):
        3 = (seq_len - patch_len) / (patch_len/2) + 2
        1 = (seq_len - patch_len) / (patch_len/2)
        patch_len/2 = seq_len - patch_len
        1.5 * patch_len = seq_len
        max_patch_len = seq_len / 1.5 = seq_len * 2/3
    
    For safety, we use seq_len // 2 as max to ensure 4+ patches.
    """
    # Calculate maximum usable patch length
    # With 50% overlap, patch_len <= seq_len * 2/3 gives ~3 patches
    # We use seq_len // 2 for safety (ensures 4+ patches)
    max_patch = seq_len // 2
    
    print(f"\n  ℹ️  Max usable patch length: {max_patch} days (seq_len={seq_len})")
    print(f"      Patches > {max_patch} days would yield < {min_patches_required} patches")
    
    if not dominant_periods:
        # Fallback to proportional method
        return [seq_len // 12, seq_len // 6, seq_len // 3]
    
    # Show which periods are excluded
    excluded = [(p, m) for p, m in dominant_periods if p > max_patch]
    if excluded:
        print(f"\n  ⚠️  Excluded dominant periods (too large for seq_len={seq_len}):")
        for p, m in excluded[:5]:
            num_patches_if_used = int((seq_len - p) / (p // 2) + 2)
            print(f"      • {p} days (magnitude: {m:.2f}) → would only give ~{num_patches_if_used} patches")
        print(f"\n      💡 TIP: Increase seq_len to 90-120 days to use these periods")
    
    # Get unique periods within valid range
    valid_periods = [p for p, _ in dominant_periods if 3 <= p <= max_patch]
    valid_periods = sorted(set(valid_periods))
    
    if len(valid_periods) < num_scales:
        # Supplement with proportional periods
        proportional = [seq_len // 12, seq_len // 6, seq_len // 3]
        valid_periods = sorted(set(valid_periods + proportional))
    
    # Select diverse scales (short, medium, long)
    if len(valid_periods) >= num_scales:
        indices = np.linspace(0, len(valid_periods) - 1, num_scales, dtype=int)
        selected = [valid_periods[i] for i in indices]
    else:
        selected = valid_periods[:num_scales]
    
    return sorted(selected)


def calculate_strides(patch_lens, overlap_ratio=0.5):
    """Calculate strides with given overlap ratio."""
    return [max(1, int(pl * (1 - overlap_ratio))) for pl in patch_lens]


def visualize_analysis(df, prices, detrended, frequencies, magnitudes, periods,
                       dominant_periods, recommended_patches, strides, seq_len=60):
    """Create comprehensive visualization of the analysis."""
    
    fig = plt.figure(figsize=(16, 14))
    
    # Color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#C73E1D',
        'dark': '#3C3C3C'
    }
    
    # 1. Original Price Series
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(df['Date'], prices, color=colors['primary'], linewidth=0.8, alpha=0.9)
    ax1.fill_between(df['Date'], prices.min(), prices, alpha=0.2, color=colors['primary'])
    ax1.set_title('VCB Stock Price (Adj Close)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (VND)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Detrended Data
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(detrended, color=colors['secondary'], linewidth=0.6, alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Detrended Price (Cyclical Component)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Deviation from Trend')
    
    # 3. Frequency Spectrum
    ax3 = fig.add_subplot(3, 2, 3)
    valid_idx = (periods > 2) & (periods < 200) & np.isfinite(periods)
    ax3.semilogy(periods[valid_idx], magnitudes[valid_idx], 
                 color=colors['primary'], linewidth=0.8, alpha=0.7)
    
    # Mark dominant periods
    for i, (period, mag) in enumerate(dominant_periods[:5]):
        color = colors['accent'] if i < 3 else colors['success']
        ax3.axvline(x=period, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        ax3.annotate(f'{period}d', xy=(period, mag), xytext=(period+5, mag*1.5),
                    fontsize=9, color=color, fontweight='bold')
    
    ax3.set_title('Frequency Spectrum (Log Scale)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Period (Trading Days)')
    ax3.set_ylabel('Magnitude (log)')
    ax3.set_xlim(2, 150)
    ax3.invert_xaxis()
    
    # 4. Top Dominant Periods Bar Chart
    ax4 = fig.add_subplot(3, 2, 4)
    top_n = min(8, len(dominant_periods))
    periods_list = [p for p, _ in dominant_periods[:top_n]]
    mags_list = [m for _, m in dominant_periods[:top_n]]
    
    bar_colors = [colors['accent'] if i < 3 else colors['primary'] for i in range(top_n)]
    bars = ax4.barh(range(top_n), mags_list, color=bar_colors, alpha=0.8)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels([f'{p} days' for p in periods_list])
    ax4.set_title('Top Dominant Periods (Recommended in Orange)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Magnitude')
    ax4.invert_yaxis()
    
    # Add trading interpretation
    interpretations = {
        5: 'Weekly',
        10: 'Bi-weekly',
        21: 'Monthly',
        42: '2-Month',
        63: 'Quarterly',
        126: 'Semi-annual',
        252: 'Annual'
    }
    for i, p in enumerate(periods_list):
        closest = min(interpretations.keys(), key=lambda x: abs(x - p))
        if abs(closest - p) <= 3:
            ax4.annotate(f'  ~{interpretations[closest]}', xy=(mags_list[i], i),
                        fontsize=8, va='center', color='gray')
    
    # 5. Recommended Patch Configurations
    ax5 = fig.add_subplot(3, 2, 5)
    
    # Show a sample window with patch overlays
    sample_start = len(prices) - seq_len - 50
    sample_data = prices[sample_start:sample_start + seq_len]
    x = np.arange(seq_len)
    
    ax5.plot(x, sample_data, color=colors['dark'], linewidth=1.5, label='Sample Window')
    
    patch_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (pl, st) in enumerate(zip(recommended_patches, strides)):
        # Draw patch regions
        for start in range(0, seq_len - pl + 1, st):
            if start == 0:
                ax5.axvspan(start, start + pl, alpha=0.15, color=patch_colors[i],
                           label=f'Patch {pl}d (stride={st})')
            else:
                ax5.axvspan(start, start + pl, alpha=0.15, color=patch_colors[i])
    
    ax5.set_title(f'Multi-Scale Patching Visualization (seq_len={seq_len})', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('Days in Window')
    ax5.set_ylabel('Price')
    ax5.legend(loc='upper right', fontsize=9)
    
    # 6. Summary Table
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
    ═══════════════════════════════════════════════════════
                    FFT ANALYSIS SUMMARY
    ═══════════════════════════════════════════════════════
    
    📊 Data Overview:
       • Total trading days: {len(prices)}
       • Analysis period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
       • Sequence length: {seq_len} days
    
    🔍 Top Discovered Periods:
    """
    for i, (period, mag) in enumerate(dominant_periods[:5], 1):
        closest = min(interpretations.keys(), key=lambda x: abs(x - period))
        interp = interpretations.get(closest, '') if abs(closest - period) <= 5 else ''
        summary_text += f"       {i}. {period:3d} days (magnitude: {mag:.4f}) {interp}\n"
    
    summary_text += f"""
    ✅ RECOMMENDED PATCH CONFIGURATION:
       ┌────────────────────────────────────────────┐
       │  patch_lens = {recommended_patches}
       │  strides    = {strides}
       └────────────────────────────────────────────┘
    
    📝 Code to use in TimeLLM_Stock_V3.py:
       
       self.patch_lens = {recommended_patches}
       self.strides = {strides}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('patch_length_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\n✅ Visualization saved to: patch_length_analysis.png")


def compute_patch_stats(prices, patch_lens, strides, seq_len=60):
    """Compute statistics about patch coverage."""
    print("\n📊 Patch Configuration Statistics:")
    print("=" * 50)
    
    total_patches = 0
    for pl, st in zip(patch_lens, strides):
        num_patches = int((seq_len - pl) / st + 2)  # +2 for padding
        overlap_pct = (1 - st/pl) * 100
        total_patches += num_patches
        print(f"  Patch {pl:2d}d (stride {st:2d}): {num_patches:3d} patches, {overlap_pct:.0f}% overlap")
    
    print("-" * 50)
    print(f"  Total patches per sample: {total_patches}")
    print("=" * 50)
    
    return total_patches


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("  FFT-Based Patch Length Analysis for VCB Stock Data")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = 'gia_VCB_2020-2025.csv'
    SEQ_LEN = 120  # Increased to capture 40-50 day patterns
    NUM_SCALES = 3  # Number of patch scales
    
    # Load data
    print("\n📁 Loading data...")
    df = load_and_preprocess_data(DATA_PATH)
    prices = df['Adj Close'].values
    
    # Detrend
    print("\n📈 Detrending data...")
    detrended, trend = detrend_data(prices)
    
    # Compute FFT
    print("\n🔬 Computing FFT spectrum...")
    frequencies, magnitudes, periods = compute_fft_spectrum(detrended)
    
    # Find dominant periods
    print("\n🔍 Finding dominant periods...")
    dominant_periods = find_dominant_periods(
        frequencies, magnitudes, periods,
        min_period=3,
        max_period=SEQ_LEN,  # Max period = sequence length
        num_peaks=10
    )
    
    print("\n  Top 10 Dominant Periods:")
    for i, (period, mag) in enumerate(dominant_periods, 1):
        print(f"    {i:2d}. {period:3d} days  (magnitude: {mag:.6f})")
    
    # Recommend patch lengths
    print("\n🎯 Recommending patch lengths...")
    recommended_patches = recommend_patch_lengths(dominant_periods, SEQ_LEN, NUM_SCALES)
    strides = calculate_strides(recommended_patches, overlap_ratio=0.5)
    
    print(f"\n  ✅ Recommended Configuration:")
    print(f"     patch_lens = {recommended_patches}")
    print(f"     strides    = {strides}")
    
    # Compute stats
    compute_patch_stats(prices, recommended_patches, strides, SEQ_LEN)
    
    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_analysis(
        df, prices, detrended, frequencies, magnitudes, periods,
        dominant_periods, recommended_patches, strides, SEQ_LEN
    )
    
    # Return config for use in model
    return {
        'patch_lens': recommended_patches,
        'strides': strides,
        'dominant_periods': dominant_periods
    }


if __name__ == '__main__':
    config = main()
    
    print("\n" + "=" * 60)
    print("  COPY THIS TO YOUR MODEL:")
    print("=" * 60)
    print(f"""
# In TimeLLM_Stock_V3.py __init__:
if self.use_multi_scale:
    # FFT-discovered optimal patch lengths for VCB stock
    self.patch_lens = {config['patch_lens']}
    self.strides = {config['strides']}
    self.patch_nums_list = [
        int((configs.seq_len - pl) / st + 2) 
        for pl, st in zip(self.patch_lens, self.strides)
    ]
    self.total_patch_nums = sum(self.patch_nums_list)
""")

