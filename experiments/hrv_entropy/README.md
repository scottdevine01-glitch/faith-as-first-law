```markdown
# Predictions B2 & M1: HRV Entropy and Crystal Amplification

**Hypothesis B2:** Heart rate variability during faithful states (compassion, gratitude) shows lower sample entropy than during stress or neutral tasks.

**Hypothesis M1:** Meditation with a quartz crystal increases HRV coherence more than with a placebo stone.

## Protocol Summary
- **Design:** Within-subjects crossover, double-blinded
- **Conditions:**
  1. Compassion meditation (loving-kindness)
  2. Resentment recall (stress induction)
  3. Neutral control (audiobook listening)
  4. *With/without quartz crystal (for M1)*
- **Measurement:** ECG recording → R-R interval extraction
- **Metrics:**
  - Sample Entropy (SampEn)
  - LF/HF ratio (coherence)
  - SDNN, RMSSD (time domain)
- **Analysis:** Compare entropy measures across conditions

## Expected Results
- **B2:** SampEn(compassion) < SampEn(neutral) < SampEn(resentment)
- **M1:** ΔSampEn(crystal) < ΔSampEn(placebo)

## Quick Start

```bash
# Run with example data
python hrv_analysis.py --example

# Analyze your ECG data
python hrv_analysis.py --ecg_dir ./ecg_data/ --output results/
```

### Files

- hrv_analysis.py – Main HRV analysis script
- crystal_study.py – Crystal amplification study protocol
- data/ – ECG data templates and examples
- results/ – Output directory

### Data Collection

1. Equipment:
   - FDA-cleared HRV monitor (Polar H10, Empatica E4, etc.)
   - ECG recording software (Kubios, AcqKnowledge, LabChart)
   - Quartz crystal and identical placebo stone
2. Procedure:
   - 5-minute baseline recording (resting)
   - 10-minute intervention (randomized condition)
   - 5-minute recovery
   - Counterbalance conditions across sessions
3. Data Format:
   - Raw ECG signal (1000 Hz sampling)
   - R-peak detection → R-R intervals (in seconds)
   - Save as: P001_baseline.csv, P001_compassion.csv, etc.
4. Blinding:
   - Third-party randomizes crystal/placebo
   - Opaque bags for stones
   - Participant unaware of condition

### Analysis

Run full HRV analysis:

```bash
# For Prediction B2
python hrv_analysis.py --data ./participant_data/ --conditions compassion,resentment,neutral --output hrv_results/

# For Prediction M1
python crystal_study.py --crystal_data ./crystal_trials/ --placebo_data ./placebo_trials/ --output crystal_results/
```

### Output

- hrv_metrics.csv – All HRV metrics per condition
- entropy_comparison.csv – Sample entropy results
- statistical_tests.csv – Hypothesis tests
- hrv_plots.png – Visualizations
- analysis_report.json – Complete results

### Dependencies

- Python 3.9+
- heartpy, neurokit2, wfdb
- pandas, numpy, scipy, matplotlib
- Optional: Kubios HRV software for validation

### Ethics Note

- Informed consent for physiological recording
- Debriefing after stress conditions
- Option to withdraw at any time
- Secure storage of physiological data
- IRB approval for academic institutions

```

---

## **File: `experiments/hrv_entropy/hrv_analysis.py`**

```python
#!/usr/bin/env python3
"""
HRV Entropy Analysis
Predictions B2 & M1: Analyze heart rate variability entropy across moral states.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import specialized HRV libraries
try:
    import heartpy as hp
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False
    print("Warning: heartpy not installed. Install with: pip install heartpy")

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    print("Warning: neurokit2 not installed. Install with: pip install neurokit2")

class HRVAnalyzer:
    """Analyze heart rate variability and entropy."""
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.results = {}
    
    def load_rr_intervals(self, filepath: Path) -> np.ndarray:
        """Load R-R intervals from file."""
        if filepath.suffix == '.csv':
            data = pd.read_csv(filepath)
            # Try common column names
            for col in ['rr', 'RR', 'interval', 'RRI', 'rr_intervals']:
                if col in data.columns:
                    return data[col].values
            
            # If no specific column found, use first column
            return data.iloc[:, 0].values
        
        elif filepath.suffix == '.txt':
            return np.loadtxt(filepath)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def clean_rr_intervals(self, rr_intervals: np.ndarray, 
                          method: str = 'kubios') -> np.ndarray:
        """Clean R-R intervals by removing artifacts."""
        rr_ms = rr_intervals * 1000  # Convert to milliseconds
        
        if method == 'kubios':
            # Kubios method: remove intervals differing >20% from previous
            cleaned = []
            for i in range(len(rr_ms)):
                if i == 0:
                    cleaned.append(rr_ms[i])
                else:
                    diff = abs(rr_ms[i] - rr_ms[i-1])
                    if diff / rr_ms[i-1] < 0.2:  # Less than 20% difference
                        cleaned.append(rr_ms[i])
                    else:
                        # Interpolate with previous value
                        cleaned.append(rr_ms[i-1])
            return np.array(cleaned) / 1000  # Back to seconds
        
        elif method == 'heartpy':
            if HEARTPY_AVAILABLE:
                cleaned, _ = hp.preprocessing.interpolate_clipping(rr_ms)
                return cleaned / 1000
            else:
                return rr_intervals
        
        else:
            return rr_intervals
    
    def calculate_time_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate time domain HRV metrics."""
        rr_ms = rr_intervals * 1000  # Work in milliseconds
        
        metrics = {
            'mean_rr': np.mean(rr_ms),
            'std_rr': np.std(rr_ms, ddof=1),  # SDNN
            'rmssd': np.sqrt(np.mean(np.diff(rr_ms) ** 2)),
            'nn50': np.sum(np.abs(np.diff(rr_ms)) > 50),
            'pnn50': (np.sum(np.abs(np.diff(rr_ms)) > 50) / len(np.diff(rr_ms))) * 100,
            'min_rr': np.min(rr_ms),
            'max_rr': np.max(rr_ms),
            'range_rr': np.max(rr_ms) - np.min(rr_ms)
        }
        
        return metrics
    
    def calculate_frequency_domain(self, rr_intervals: np.ndarray, 
                                  method: str = 'welch') -> Dict[str, float]:
        """Calculate frequency domain HRV metrics."""
        # Interpolate to evenly spaced time series
        t = np.cumsum(rr_intervals)
        t_interp = np.arange(t[0], t[-1], 1/self.sampling_rate)
        rr_interp = np.interp(t_interp, t, rr_intervals)
        
        # Remove linear trend
        rr_detrended = signal.detrend(rr_interp)
        
        if method == 'welch':
            # Welch's method for PSD
            freqs, psd = signal.welch(rr_detrended, fs=self.sampling_rate, 
                                     nperseg=min(256, len(rr_detrended)//4))
        
        elif method == 'lomb':
            # Lomb-Scargle for unevenly spaced data
            from astropy.timeseries import LombScargle
            ls = LombScargle(t, rr_intervals)
            freqs, psd = ls.autopower(minimum_frequency=0.01, maximum_frequency=0.5)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Define frequency bands (Hz)
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # Calculate power in each band
        vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])], 
                            freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
        lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], 
                           freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], 
                           freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        total_power = vlf_power + lf_power + hf_power
        
        metrics = {
            'total_power': total_power,
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else np.nan,
            'lf_nu': (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else np.nan,
            'hf_nu': (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else np.nan,
            'peak_lf': freqs[np.argmax(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])] if lf_power > 0 else np.nan,
            'peak_hf': freqs[np.argmax(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])] if hf_power > 0 else np.nan
        }
        
        return metrics
    
    def calculate_sample_entropy(self, rr_intervals: np.ndarray, 
                                m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Sample Entropy (SampEn).
        
        Args:
            rr_intervals: R-R interval time series
            m: Template length (usually 2)
            r: Tolerance (usually 0.2 * standard deviation)
            
        Returns:
            Sample entropy value
        """
        n = len(rr_intervals)
        
        if n <= m:
            return np.nan
        
        # Convert to numpy array
        data = np.asarray(rr_intervals)
        
        # Calculate standard deviation
        std = np.std(data, ddof=1)
        if std == 0:
            return np.nan
        
        # Set tolerance
        tolerance = r * std
        
        # Pre-compute distances
        def _maxdist(xi, xj):
            return max(abs(xi[k] - xj[k]) for k in range(m))
        
        # Count matches for m and m+1
        B = 0.0
        A = 0.0
        
        # For each template of length m
        for i in range(n - m):
            xi = data[i:i + m]
            
            # Compare with all other templates
            for j in range(i + 1, n - m):  # j > i to avoid counting self-matches twice
                xj = data[j:j + m]
                
                if _maxdist(xi, xj) <= tolerance:
                    B += 1
                    
                    # Check for m+1 length
                    if abs(data[i + m] - data[j + m]) <= tolerance:
                        A += 1
        
        # Avoid division by zero
        if B == 0:
            return np.nan
        
        # Calculate Sample Entropy
        sampen = -np.log(A / B) if A > 0 else -np.log(1 / (B * (n - m)))
        
        return sampen
    
    def calculate_approximate_entropy(self, rr_intervals: np.ndarray, 
                                     m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Approximate Entropy (ApEn).
        """
        n = len(rr_intervals)
        
        if n <= m:
            return np.nan
        
        data = np.asarray(rr_intervals)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return np.nan
        
        tolerance = r * std
        
        def _phi(m_val):
            """Helper function for ApEn calculation."""
            patterns = np.lib.stride_tricks.sliding_window_view(data, m_val)
            n_patterns = len(patterns)
            
            C = np.zeros(n_patterns)
            for i in range(n_patterns):
                # Count number of similar patterns
                diff = np.abs(patterns - patterns[i])
                max_diff = np.max(diff, axis=1)
                C[i] = np.sum(max_diff <= tolerance) / n_patterns
            
            # Return average of log(C)
            return np.sum(np.log(C[C > 0])) / n_patterns
        
        # Calculate ApEn
        apen = _phi(m) - _phi(m + 1)
        
        return apen
    
    def calculate_poincare(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate Poincaré plot metrics."""
        rr_n = rr_intervals[:-1]  # RR_n
        rr_n1 = rr_intervals[1:]  # RR_{n+1}
        
        # Ellipse fitting
        sd1 = np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2)
        sd2 = np.std(rr_n + rr_n1, ddof=1) / np.sqrt(2)
        
        metrics = {
            'sd1': sd1,
            'sd2': sd2,
            'sd1_sd2_ratio': sd1 / sd2 if sd2 != 0 else np.nan,
            'ellipse_area': np.pi * sd1 * sd2,
            'poincare_mean': np.mean(np.sqrt((rr_n - np.mean(rr_n))**2 + (rr_n1 - np.mean(rr_n1))**2))
        }
        
        return metrics
    
    def calculate_all_metrics(self, rr_intervals: np.ndarray, 
                             participant_id: str, condition: str) -> Dict[str, any]:
        """Calculate all HRV metrics for a given recording."""
        # Clean data
        cleaned_rr = self.clean_rr_intervals(rr_intervals)
        
        results = {
            'participant_id': participant_id,
            'condition': condition,
            'recording_length': len(cleaned_rr),
            'duration_minutes': np.sum(cleaned_rr) / 60,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Time domain metrics
        results.update(self.calculate_time_domain(cleaned_rr))
        
        # Frequency domain metrics (if enough data)
        if len(cleaned_rr) >= 120:  # Need at least 2 minutes for reliable spectral analysis
            try:
                results.update(self.calculate_frequency_domain(cleaned_rr))
            except:
                pass
        
        # Entropy measures
        if len(cleaned_rr) >= 100:  # Need sufficient data for entropy
            results['sample_entropy'] = self.calculate_sample_entropy(cleaned_rr)
            results['approximate_entropy'] = self.calculate_approximate_entropy(cleaned_rr)
        
        # Poincaré plot metrics
        if len(cleaned_rr) >= 10:
            results.update(self.calculate_poincare(cleaned_rr))
        
        # Additional complexity measures
        if len(cleaned_rr) >= 50:
            results['hurst_exponent'] = self.calculate_hurst_exponent(cleaned_rr)
            results['detrended_fluctuation'] = self.calculate_dfa(cleaned_rr)
        
        return results
    
    def calculate_hurst_exponent(self, rr_intervals: np.ndarray) -> float:
        """Calculate Hurst exponent (long-range correlations)."""
        n = len(rr_intervals)
        if n < 50:
            return np.nan
        
        # Rescaled range analysis
        lags = range(2, n//4)
        tau = []
        rs = []
        
        for lag in lags:
            # Create subsets
            k = n // lag
            r_s = []
            
            for i in range(k):
                subset = rr_intervals[i*lag:(i+1)*lag]
                mean_subset = np.mean(subset)
                cum_dev = np.cumsum(subset - mean_subset)
                r = np.max(cum_dev) - np.min(cum_dev)
                s = np.std(subset, ddof=1)
                if s > 0:
                    r_s.append(r / s)
            
            if r_s:
                tau.append(lag)
                rs.append(np.mean(r_s))
        
        if len(tau) > 1:
            # Fit log(R/S) vs log(tau)
            log_tau = np.log(tau)
            log_rs = np.log(rs)
            slope, _, _, _, _ = stats.linregress(log_tau, log_rs)
            return slope
        else:
            return np.nan
    
    def calculate_dfa(self, rr_intervals: np.ndarray) -> float:
        """Calculate Detrended Fluctuation Analysis (DFA) alpha exponent."""
        n = len(rr_intervals)
        if n < 50:
            return np.nan
        
        # Integrated series
        y = np.cumsum(rr_intervals - np.mean(rr_intervals))
        
        # Window sizes
        window_sizes = np.unique(np.logspace(np.log10(4), np.log10(n//4), 20).astype(int))
        
        f = []
        
        for window in window_sizes:
            # Break into windows
            n_windows = n // window
            if n_windows < 4:
                continue
            
            rms = []
            
            for i in range(n_windows):
                segment = y[i*window:(i+1)*window]
                x = np.arange(len(segment))
                
                # Linear detrending
                slope, intercept, _, _, _ = stats.linregress(x, segment)
                trend = slope * x + intercept
                detrended = segment - trend
                
                rms.append(np.sqrt(np.mean(detrended**2)))
            
            f.append(np.mean(rms))
        
        if len(f) > 1:
            # Fit log(F) vs log(window)
            log_w = np.log(window_sizes[:len(f)])
            log_f = np.log(f)
            slope, _, _, _, _ = stats.linregress(log_w, log_f)
            return slope
        else:
            return np.nan

def generate_example_data() -> Dict[str, np.ndarray]:
    """Generate example RR interval data for different conditions."""
    np.random.seed(42)
    
    # Base parameters
    n_points = 300  # ~5 minutes at average 1-second intervals
    base_rr = 0.8  # 800ms average RR
    
    # Compassion: regular, coherent pattern
    t = np.linspace(0, 10, n_points)
    compassion_rr = base_rr + 0.05 * np.sin(2*np.pi*0.1*t)  # 0.1 Hz oscillation
    compassion_rr += np.random.normal(0, 0.02, n_points)  # Small noise
    
    # Neutral: moderately variable
    neutral_rr = base_rr + np.random.normal(0, 0.05, n_points)
    
    # Resentment: chaotic, less coherent
    resentment_rr = base_rr + 0.1 * np.random.randn(n_points)
    # Add some abrupt changes
    change_points = np.random.choice(n_points, 10, replace=False)
    for cp in change_points:
        resentment_rr[cp:] += np.random.choice([-0.1, 0.1])
    
    # Ensure positive values
    compassion_rr = np.abs(compassion_rr)
    neutral_rr = np.abs(neutral_rr)
    resentment_rr = np.abs(resentment_rr)
    
    return {
        'compassion': compassion_rr,
        'neutral': neutral_rr,
        'resentment': resentment_rr
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze HRV entropy across moral states')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing RR interval files')
    parser.add_argument('--example', action='store_true',
                       help='Generate and analyze example data')
    parser.add_argument('--output', type=str, default='hrv_results',
                       help='Output directory for results')
    parser.add_argument('--conditions', type=str, default='compassion,neutral,resentment',
                       help='Comma-separated list of conditions to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    analyzer = HRVAnalyzer(sampling_rate=1000.0)
    
    if args.example:
        print("="*60)
        print("HRV ENTROPY ANALYSIS - EXAMPLE")
        print("="*60)
        
        # Generate example data
        example_data = generate_example_data()
        
        # Save example data
        for condition, rr in example_data.items():
            df = pd.DataFrame({'rr_interval': rr})
            df.to_csv(output_dir / f'example_{condition}.csv', index=False)
        
        print(f"✓ Example data saved to {output_dir}")
        
        # Analyze each condition
        print("\nAnalyzing HRV metrics...")
        all_results = []
        
        for condition, rr in example_data.items():
            results = analyzer.calculate_all_metrics(rr, 'example', condition)
            all_results.append(results)
            
            print(f"\n{condition.upper()}:")
            print(f"  Sample Entropy: {results.get('sample_entropy', 'N/A'):.3f}")
            print(f"  SDNN: {results.get('std_rr', 'N/A'):.1f} ms")
            print(f"  RMSSD: {results.get('rmssd', 'N/A'):.1f} ms")
            if 'lf_hf_ratio' in results:
                print(f"  LF/HF ratio: {results.get('lf_hf_ratio', 'N/A'):.2f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'hrv_metrics.csv', index=False)
        
        # Hypothesis check
        print("\n" + "="*60)
        print("HYPOTHESIS CHECK")
        print("="*60)
        
        # Get sample entropy values
        sampen_values = {}
        for condition in ['compassion', 'neutral', 'resentment']:
            condition_data = results_df[results_df['condition'] == condition]
            if len(condition_data) > 0:
                sampen_values[condition] = condition_data.iloc[0].get('sample_entropy', np.nan)
        
        print(f"\nPredicted pattern: compassion < neutral < resentment")
        print(f"  SampEn(compassion): {sampen_values.get('compassion', 'N/A'):.3f}")
        print(f"  SampEn(neutral): {sampen_values.get('neutral', 'N/A'):.3f}")
        print(f"  SampEn(resentment): {sampen_values.get('resentment', 'N/A'):.3f}")
        
        # Check inequalities
        if all(not np.isnan(v) for v in sampen_values.values()):
            pattern_holds = (sampen_values['compassion'] < sampen_values['neutral'] < 
                           sampen_values['resentment'])
            
            if pattern_holds:
                print("  ✓ Hypothesis B2 supported: Entropy increases with stress")
            else:
                print("  ✗ Hypothesis B2 not supported")
        
        print(f"\nDetailed results saved to: {output_dir / 'hrv_metrics.csv'}")
    
    elif args.data_dir:
        print("="*60)
        print("HRV ENTROPY ANALYSIS")
        print("="*60)
        
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"Error: Directory {data_path} not found")
            return
        
        # Parse conditions
        conditions = [c.strip() for c in args.conditions.split(',')]
        
        # Find RR interval files
        all_results = []
        
        for condition in conditions:
            pattern = f"*{condition}*.csv"
            files = list(data_path.glob(pattern)) + list(data_path.glob(pattern.replace('csv', 'txt')))
            
            for file_path in files:
                # Extract participant ID from filename
                filename = file_path.stem
                participant_id = filename.split('_')[0] if '_' in filename else 'unknown'
                
                print(f"Processing: {filename}")
                
                try:
                    # Load RR intervals
                    rr_intervals = analyzer.load_rr_intervals(file_path)
                    
                    # Analyze
                    results = analyzer.calculate_all_metrics(rr_intervals, participant_id, condition)
                    all_results.append(results)
                    
                    print(f"  ✓ {len(rr_intervals)} intervals, SampEn: {results.get('sample_entropy', 'N/A'):.3f}")
                    
                except Exception as e:
                    print(f"  ✗ Error analyzing {filename}: {e}")
        
        if not all_results:
            print("No valid data files found")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_file = output_dir / 'hrv_analysis_results.csv'
        results_df.to_csv(results_file, index=False)
        
        # Summary statistics by condition
        print("\n" + "="*60)
        print("SUMMARY STATISTICS BY CONDITION")
        print("="*60)
        
        summary = results_df.groupby('condition').agg({
            'sample_entropy': ['mean', 'std', 'count'],
            'std_rr': 'mean',
            'rmssd': 'mean',
            'lf_hf_ratio': 'mean'
        }).round(3)
        
        print(summary.to_string())
        
        # Save summary
        summary.to_csv(output_dir / 'summary_statistics.csv')
        
        # Hypothesis test
        print("\n" + "="*60)
        print("HYPOTHESIS TEST")
        print("="*60)
        
        # Check if we have enough conditions
        available_conditions = results_df['condition'].unique()
        if len(available_conditions) >= 2:
            # Perform statistical tests
            from scipy.stats import f_oneway, kruskal
            
            # Group data by condition
            groups = []
            condition_names = []
            
            for condition in available_conditions:
                condition_data = results_df[results_df['condition'] == condition]['sample_entropy'].values
                if len(condition_data) > 0:
                    groups.append(condition_data)
                    condition_names.append(condition)
            
            if len(groups) >= 2:
                # ANOVA or Kruskal-Wallis
                if all(len(g) >= 5 for g in groups) and len(groups) <= 3:
                    # Parametric test
                    f_stat, p_value = f_oneway(*groups)
                    test_name = "One-way ANOVA"
                else:
                    # Non-parametric test
                    h_stat, p_value = kruskal(*groups)
                    test_name = "Kruskal-Wallis"
                
                print(f"{test_name}:")
                print(f"  p-value = {p_value:.6f}")
                
                if p_value < 0.05:
                    print("  ✓ Significant difference between conditions")
                    
                    # Post-hoc comparisons
                    print("\nPost-hoc comparisons:")
                    from itertools import combinations
                    for i, j in combinations(range(len(groups)), 2):
                        cond1, cond2 = condition_names[i], condition_names[j]
                        t_stat, p_val = stats.ttest_ind(groups[i], groups[j], equal_var=False)
                        print(f"  {cond1} vs {cond2}: p = {p_val:.4f}")
                else:
                    print("  ✗ No significant difference between conditions")
            
            # Check hypothesis pattern
            if set(['compassion', 'neutral', 'resentment']).issubset(set(available_conditions)):
                mean_sampen = results_df.groupby('condition')['sample_entropy'].mean()
                
                print(f"\nPredicted pattern: compassion < neutral < resentment")
                print(f"  Actual pattern: ", end="")
                for cond in ['compassion', 'neutral', 'resentment']:
                    print(f"{cond}={mean_sampen.get(cond, 'N/A'):.3f} ", end="")
                print()
                
                if (mean_sampen.get('compassion', np.inf) < 
                    mean_sampen.get('neutral', np.inf) < 
                    mean_sampen.get('resentment', -np.inf)):
                    print("  ✓ Hypothesis B2 supported")
                else:
                    print("  ✗ Hypothesis B2 not supported")
        
        print(f"\n✓ Analysis complete!")
        print(f"  Results saved to: {results_file}")
        print(f"  Summary saved to: {output_dir / 'summary_statistics.csv'}")
    
    else:
        print("Please provide --data_dir or use --example")
        parser.print_help()

if __name__ == '__main__':
    main()
```
