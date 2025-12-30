#!/usr/bin/env python3
"""
Process HRV data for Predictions B2 & M1.
Extracts HRV features from R-R interval data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import neurokit2 as nk
    import heartpy as hp
except ImportError:
    print("Warning: neurokit2 and heartpy not installed. Install with: pip install neurokit2 heartpy")

def load_rr_data(filepath: str, format: str = 'csv') -> pd.DataFrame:
    """
    Load R-R interval data from various formats.
    
    Args:
        filepath: Path to data file
        format: File format ('csv', 'txt', 'edf', 'wfdb')
        
    Returns:
        DataFrame with R-R intervals in seconds
    """
    if format == 'csv':
        df = pd.read_csv(filepath)
        
        # Try to identify R-R column
        rr_columns = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['rr', 'r-r', 'interval', 'ibi'])]
        
        if rr_columns:
            rr_data = df[rr_columns[0]].values
        else:
            # Assume first numeric column is R-R
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                rr_data = df[numeric_cols[0]].values
            else:
                raise ValueError("Could not find R-R interval data in CSV")
    
    elif format == 'txt':
        # Simple text file with one R-R value per line
        rr_data = np.loadtxt(filepath)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Convert to DataFrame with time index
    rr_times = np.cumsum(rr_data)  # Cumulative sum for timestamps
    df_rr = pd.DataFrame({
        'RR_intervals': rr_data,
        'Time': rr_times
    })
    
    return df_rr

def clean_rr_data(rr_intervals: np.ndarray, 
                  sample_rate: float = 1000,
                  method: str = 'neurokit') -> np.ndarray:
    """
    Clean R-R interval data (remove artifacts, ectopic beats).
    
    Args:
        rr_intervals: Raw R-R intervals in seconds
        sample_rate: Original sampling rate in Hz
        method: Cleaning method ('neurokit', 'heartpy', 'custom')
        
    Returns:
        Cleaned R-R intervals
    """
    if method == 'neurokit':
        # Use neurokit2 for cleaning
        rr_cleaned = nk.ppg_intervalrelated(rr_intervals, sampling_rate=sample_rate)
        return rr_cleaned
    
    elif method == 'heartpy':
        # Use heartpy for cleaning
        working_data, measures = hp.process(rr_intervals, sample_rate)
        return working_data['RR_list']
    
    elif method == 'custom':
        # Custom cleaning: remove outliers beyond 3 SD
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        # Filter based on physiological limits (0.3-2.0 seconds)
        valid_mask = (rr_intervals > 0.3) & (rr_intervals < 2.0)
        
        # Remove extreme outliers
        outlier_mask = np.abs(rr_intervals - mean_rr) < 3 * std_rr
        
        cleaned = rr_intervals[valid_mask & outlier_mask]
        return cleaned
    
    else:
        raise ValueError(f"Unknown cleaning method: {method}")

def calculate_time_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Calculate time-domain HRV features.
    
    Args:
        rr_intervals: Cleaned R-R intervals in seconds
        
    Returns:
        Dictionary of time-domain features
    """
    # Convert to milliseconds for standard HRV metrics
    rr_ms = rr_intervals * 1000
    
    features = {
        'mean_rr': np.mean(rr_ms),
        'sdnn': np.std(rr_ms),  # Standard deviation of NN intervals
        'rmssd': np.sqrt(np.mean(np.diff(rr_ms) ** 2)),  # Root mean square of successive differences
        'nn50': np.sum(np.abs(np.diff(rr_ms)) > 50),  # Number of pairs differing by >50ms
        'pnn50': (np.sum(np.abs(np.diff(rr_ms)) > 50) / len(np.diff(rr_ms))) * 100,
        'min_rr': np.min(rr_ms),
        'max_rr': np.max(rr_ms),
        'range_rr': np.max(rr_ms) - np.min(rr_ms)
    }
    
    return features

def calculate_frequency_domain_features(rr_intervals: np.ndarray, 
                                       sampling_rate: float = 4) -> Dict[str, float]:
    """
    Calculate frequency-domain HRV features using Lomb-Scargle periodogram.
    
    Args:
        rr_intervals: R-R intervals in seconds
        sampling_rate: Resampling rate for spectral analysis
        
    Returns:
        Dictionary of frequency-domain features
    """
    try:
        # Use neurokit2 for frequency analysis
        hrv_freq = nk.hrv_frequency(rr_intervals, sampling_rate=sampling_rate, show=False)
        
        features = {
            'lf': float(hrv_freq['HRV_LF'][0]),  # Low frequency power (0.04-0.15 Hz)
            'hf': float(hrv_freq['HRV_HF'][0]),  # High frequency power (0.15-0.4 Hz)
            'lf_hf_ratio': float(hrv_freq['HRV_LFHF'][0]),  # LF/HF ratio
            'lf_nu': float(hrv_freq['HRV_LFn'][0]),  # LF in normalized units
            'hf_nu': float(hrv_freq['HRV_HFn'][0]),  # HF in normalized units
            'total_power': float(hrv_freq['HRV_TotalPower'][0])  # Total spectral power
        }
        
    except Exception as e:
        print(f"Warning: Frequency analysis failed: {e}")
        features = {key: np.nan for key in ['lf', 'hf', 'lf_hf_ratio', 'lf_nu', 'hf_nu', 'total_power']}
    
    return features

def calculate_nonlinear_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Calculate nonlinear HRV features (entropy, Poincaré plot).
    
    Args:
        rr_intervals: R-R intervals in seconds
        
    Returns:
        Dictionary of nonlinear features
    """
    try:
        # Sample entropy
        sampen = nk.entropy_sample(rr_intervals)
        
        # Poincaré plot features (SD1, SD2)
        poincare = nk.hrv_nonlinear(rr_intervals)
        
        features = {
            'sampen': float(sampen[0]),  # Sample entropy
            'sd1': float(poincare['HRV_SD1'][0]),  # Poincaré SD1 (short-term variability)
            'sd2': float(poincare['HRV_SD2'][0]),  # Poincaré SD2 (long-term variability)
            'sd1_sd2_ratio': float(poincare['HRV_SD1SD2'][0])  # SD1/SD2 ratio
        }
        
    except Exception as e:
        print(f"Warning: Nonlinear analysis failed: {e}")
        features = {key: np.nan for key in ['sampen', 'sd1', 'sd2', 'sd1_sd2_ratio']}
    
    return features

def calculate_hrv_coherence(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Calculate HRV coherence metrics (similar to HeartMath).
    
    Args:
        rr_intervals: R-R intervals in seconds
        
    Returns:
        Dictionary of coherence features
    """
    try:
        # Resample to 10 Hz for coherence analysis
        from scipy import interpolate
        
        # Create time vector
        time_points = np.cumsum(rr_intervals)
        
        # Interpolate to regular time grid
        f = interpolate.interp1d(time_points, rr_intervals, kind='cubic', 
                                 bounds_error=False, fill_value='extrapolate')
        regular_time = np.arange(0, time_points[-1], 0.1)  # 10 Hz
        regular_rr = f(regular_time)
        
        # Remove NaN values
        regular_rr = regular_rr[~np.isnan(regular_rr)]
        
        # Calculate spectral coherence in respiratory band (0.1-0.4 Hz)
        from scipy import signal
        
        # Welch periodogram
        freqs, psd = signal.welch(regular_rr, fs=10, nperseg=256)
        
        # Respiratory band (0.1-0.4 Hz)
        resp_mask = (freqs >= 0.1) & (freqs <= 0.4)
        
        if np.any(resp_mask):
            resp_power = np.trapz(psd[resp_mask], freqs[resp_mask])
            total_power = np.trapz(psd, freqs)
            coherence_ratio = resp_power / total_power if total_power > 0 else 0
            
            # Peak frequency in respiratory band
            peak_idx = np.argmax(psd[resp_mask])
            peak_freq = freqs[resp_mask][peak_idx]
        else:
            coherence_ratio = 0
            peak_freq = 0
        
        features = {
            'coherence_ratio': float(coherence_ratio),
            'respiratory_peak_hz': float(peak_freq),
            'coherence_score': float(coherence_ratio * 100),  # 0-100 scale
            'coherence_category': 'low' if coherence_ratio < 0.33 else 
                                 'medium' if coherence_ratio < 0.66 else 'high'
        }
        
    except Exception as e:
        print(f"Warning: Coherence analysis failed: {e}")
        features = {key: np.nan for key in ['coherence_ratio', 'respiratory_peak_hz', 
                                           'coherence_score', 'coherence_category']}
    
    return features

def process_hrv_file(input_file: str, output_dir: Path, 
                    participant_id: str, condition: str) -> Dict[str, any]:
    """
    Process a single HRV data file and extract all features.
    
    Args:
        input_file: Path to HRV data file
        output_dir: Directory to save results
        participant_id: Participant identifier
        condition: Experimental condition
        
    Returns:
        Dictionary with all HRV features
    """
    print(f"Processing {participant_id} - {condition}...")
    
    # Load data
    try:
        df_rr = load_rr_data(input_file)
        rr_intervals = df_rr['RR_intervals'].values
        
        # Basic quality check
        if len(rr_intervals) < 30:  # Need at least 30 beats for HRV
            print(f"  Warning: Only {len(rr_intervals)} beats, minimum 30 needed")
            return None
        
        # Clean data
        rr_cleaned = clean_rr_data(rr_intervals, method='custom')
        
        # Calculate features
        time_features = calculate_time_domain_features(rr_cleaned)
        freq_features = calculate_frequency_domain_features(rr_cleaned)
        nonlinear_features = calculate_nonlinear_features(rr_cleaned)
        coherence_features = calculate_hrv_coherence(rr_cleaned)
        
        # Combine all features
        all_features = {
            'participant_id': participant_id,
            'condition': condition,
            'n_beats': len(rr_cleaned),
            'recording_duration': np.sum(rr_cleaned),
            'mean_hr': 60 / np.mean(rr_cleaned) if np.mean(rr_cleaned) > 0 else 0
        }
        all_features.update(time_features)
        all_features.update(freq_features)
        all_features.update(nonlinear_features)
        all_features.update(coherence_features)
        
        # Save individual results
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{participant_id}_{condition}_features.json"
        
        with open(output_file, 'w') as f:
            json.dump(all_features, f, indent=2)
        
        print(f"  ✓ Extracted {len(all_features)} features")
        print(f"  Saved to: {output_file}")
        
        return all_features
        
    except Exception as e:
        print(f"  ✗ Error processing {input_file}: {e}")
        return None

def batch_process_hrv(data_dir: str, output_dir: str, 
                     file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Process multiple HRV files in batch.
    
    Args:
        data_dir: Directory containing HRV files
        output_dir: Directory to save results
        file_pattern: Pattern to match HRV files
        
    Returns:
        DataFrame with all extracted features
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    all_results = []
    
    # Find all matching files
    files = list(data_path.rglob(file_pattern))
    
    if len(files) == 0:
        print(f"No files found matching {file_pattern} in {data_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(files)} HRV files to process")
    
    for file_path in files:
        # Extract participant and condition from filename
        # Expected format: P001_virtue.csv or P001_virtue_baseline.csv
        filename = file_path.stem
        
        # Try to parse participant ID and condition
        parts = filename.split('_')
        if len(parts) >= 2:
            participant_id = parts[0]
            condition = parts[1]
        else:
            print(f"  Warning: Could not parse {filename}, skipping")
            continue
        
        # Process file
        features = process_hrv_file(str(file_path), output_path / 'individual', 
                                   participant_id, condition)
        
        if features:
            all_results.append(features)
    
    # Combine all results
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Save combined results
        combined_file = output_path / 'hrv_features_combined.csv'
        df_results.to_csv(combined_file, index=False)
        
        print(f"\n✓ Processed {len(all_results)} files successfully")
        print(f"Combined results saved to: {combined_file}")
        
        # Generate summary statistics
        summary_stats = generate_summary(df_results)
        summary_file = output_path / 'hrv_summary_statistics.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Summary statistics saved to: {summary_file}")
        
        return df_results
    else:
        print("✗ No files processed successfully")
        return pd.DataFrame()

def generate_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate summary statistics for HRV features.
    
    Args:
        df: DataFrame with HRV features
        
    Returns:
        Dictionary with summary statistics
    """
    if len(df) == 0:
        return {}
    
    summary = {
        'overall': {
            'n_participants': df['participant_id'].nunique(),
            'n_observations': len(df),
            'conditions': df['condition'].unique().tolist()
        }
    }
    
    # Summary by condition
    if 'condition' in df.columns:
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            
            condition_summary = {
                'n': len(condition_data),
                'mean_sampen': float(condition_data['sampen'].mean()),
                'std_sampen': float(condition_data['sampen'].std()),
                'mean_lf_hf_ratio': float(condition_data['lf_hf_ratio'].mean()),
                'mean_coherence_score': float(condition_data['coherence_score'].mean())
            }
            
            summary[condition] = condition_summary
    
    # Check hypothesis patterns
    if all(c in df['condition'].unique() for c in ['virtue', 'neutral', 'vice']):
        virtue_mean = df[df['condition'] == 'virtue']['sampen'].mean()
        neutral_mean = df[df['condition'] == 'neutral']['sampen'].mean()
        vice_mean = df[df['condition'] == 'vice']['sampen'].mean()
        
        summary['hypothesis_check'] = {
            'sampen_virtue': float(virtue_mean),
            'sampen_neutral': float(neutral_mean),
            'sampen_vice': float(vice_mean),
            'pattern_virtue_lt_neutral': bool(virtue_mean < neutral_mean),
            'pattern_neutral_lt_vice': bool(neutral_mean < vice_mean),
            'pattern_full': bool(virtue_mean < neutral_mean < vice_mean)
        }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Process HRV data for entropy analysis')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Single file processing
    single_parser = subparsers.add_parser('single', help='Process a single HRV file')
    single_parser.add_argument('--input', type=str, required=True, help='Input HRV file')
    single_parser.add_argument('--output', type=str, required=True, help='Output directory')
    single_parser.add_argument('--participant', type=str, required=True, help='Participant ID')
    single_parser.add_argument('--condition', type=str, required=True, help='Condition')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple HRV files')
    batch_parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    batch_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    batch_parser.add_argument('--pattern', type=str, default='*.csv', help='File pattern')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        features = process_hrv_file(
            args.input, 
            Path(args.output), 
            args.participant, 
            args.condition
        )
        
        if features:
            print(f"\n✓ Successfully processed {args.participant} - {args.condition}")
            print(f"Sample entropy: {features.get('sampen', 'N/A'):.4f}")
            print(f"Coherence score: {features.get('coherence_score', 'N/A'):.1f}")
    
    elif args.command == 'batch':
        df_results = batch_process_hrv(
            args.input_dir,
            args.output_dir,
            args.pattern
        )
        
        if len(df_results) > 0:
            print("\n" + "="*60)
            print("BATCH PROCESSING COMPLETE")
            print("="*60)
            
            # Check hypothesis
            if all(c in df_results['condition'].unique() for c in ['virtue', 'neutral', 'vice']):
                virtue_mean = df_results[df_results['condition'] == 'virtue']['sampen'].mean()
                neutral_mean = df_results[df_results['condition'] == 'neutral']['sampen'].mean()
                vice_mean = df_results[df_results['condition'] == 'vice']['sampen'].mean()
                
                print(f"\nSample Entropy by Condition:")
                print(f"  Virtue:   {virtue_mean:.4f}")
                print(f"  Neutral:  {neutral_mean:.4f}")
                print(f"  Vice:     {vice_mean:.4f}")
                
                if virtue_mean < neutral_mean < vice_mean:
                    print(f"\n✓ HYPOTHESIS SUPPORTED: SampEn(virtue) < SampEn(neutral) < SampEn(vice)")
                else:
                    print(f"\n✗ HYPOTHESIS NOT SUPPORTED")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
