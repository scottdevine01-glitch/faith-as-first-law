#!/usr/bin/env python3
"""
Statistical analysis for Prediction B1: Virtuous Speech Compression
Tests whether virtuous speech compresses more than vicious speech.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load experiment results from CSV file.
    
    Args:
        filepath: Path to results CSV
        
    Returns:
        DataFrame with experiment results
    """
    df = pd.read_csv(filepath)
    
    # Ensure required columns exist
    required_cols = ['participant_id', 'condition', 'compression_ratio']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Ensure conditions are in correct order for analysis
    condition_order = ['virtue', 'neutral', 'vice']
    df['condition'] = pd.Categorical(df['condition'], categories=condition_order, ordered=True)
    
    return df.sort_values(['participant_id', 'condition'])

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each condition.
    
    Args:
        df: DataFrame with experiment data
        
    Returns:
        DataFrame with descriptive statistics
    """
    stats_df = df.groupby('condition')['compression_ratio'].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('25%', lambda x: np.percentile(x, 25)),
        ('median', 'median'),
        ('75%', lambda x: np.percentile(x, 75)),
        ('max', 'max')
    ]).round(4)
    
    # Add standard error of the mean
    stats_df['sem'] = stats_df['std'] / np.sqrt(stats_df['n'])
    
    return stats_df

def check_hypothesis_pattern(df: pd.DataFrame) -> Dict[str, any]:
    """
    Check if results follow the predicted pattern:
    C_R(virtue) < C_R(neutral) < C_R(vice)
    
    Args:
        df: DataFrame with experiment data
        
    Returns:
        Dictionary with hypothesis check results
    """
    means = df.groupby('condition')['compression_ratio'].mean()
    
    results = {
        'virtue_mean': means.get('virtue', np.nan),
        'neutral_mean': means.get('neutral', np.nan),
        'vice_mean': means.get('vice', np.nan),
        'pattern_virtue_lt_neutral': False,
        'pattern_neutral_lt_vice': False,
        'pattern_full': False
    }
    
    # Check each inequality
    if not np.isnan(results['virtue_mean']) and not np.isnan(results['neutral_mean']):
        results['pattern_virtue_lt_neutral'] = results['virtue_mean'] < results['neutral_mean']
        results['virtue_vs_neutral_diff'] = results['virtue_mean'] - results['neutral_mean']
    
    if not np.isnan(results['neutral_mean']) and not np.isnan(results['vice_mean']):
        results['pattern_neutral_lt_vice'] = results['neutral_mean'] < results['vice_mean']
        results['neutral_vs_vice_diff'] = results['neutral_mean'] - results['vice_mean']
    
    # Check full pattern
    if results['pattern_virtue_lt_neutral'] and results['pattern_neutral_lt_vice']:
        results['pattern_full'] = True
        results['virtue_vs_vice_diff'] = results['virtue_mean'] - results['vice_mean']
    
    return results

def repeated_measures_anova(df: pd.DataFrame) -> Dict[str, any]:
    """
    Perform one-way repeated measures ANOVA.
    
    Args:
        df: DataFrame with experiment data (must be in long format)
        
    Returns:
        Dictionary with ANOVA results
    """
    try:
        # For repeated measures, we need to pivot the data
        pivot_df = df.pivot(index='participant_id', columns='condition', values='compression_ratio')
        
        # Drop participants with missing data
        pivot_df = pivot_df.dropna()
        
        if len(pivot_df) < 2:
            return {'error': 'Insufficient data for ANOVA (need at least 2 complete participants)'}
        
        # Perform repeated measures ANOVA manually
        n = len(pivot_df)
        k = len(pivot_df.columns)
        
        # Calculate means
        grand_mean = pivot_df.values.mean()
        condition_means = pivot_df.mean()
        participant_means = pivot_df.mean(axis=1)
        
        # Calculate sums of squares
        ss_total = ((pivot_df.values - grand_mean) ** 2).sum()
        ss_conditions = n * ((condition_means - grand_mean) ** 2).sum()
        ss_participants = k * ((participant_means - grand_mean) ** 2).sum()
        ss_error = ss_total - ss_conditions - ss_participants
        
        # Calculate degrees of freedom
        df_conditions = k - 1
        df_participants = n - 1
        df_error = (n - 1) * (k - 1)
        
        # Calculate mean squares
        ms_conditions = ss_conditions / df_conditions
        ms_error = ss_error / df_error
        
        # Calculate F-statistic
        f_statistic = ms_conditions / ms_error
        
        # Calculate p-value
        p_value = 1 - stats.f.cdf(f_statistic, df_conditions, df_error)
        
        # Calculate effect size (partial eta squared)
        eta_squared = ss_conditions / (ss_conditions + ss_error)
        
        return {
            'f_statistic': round(f_statistic, 4),
            'df_conditions': df_conditions,
            'df_error': df_error,
            'p_value': round(p_value, 6),
            'eta_squared': round(eta_squared, 4),
            'n_complete': n,
            'conditions': list(pivot_df.columns)
        }
        
    except Exception as e:
        return {'error': f'ANOVA failed: {str(e)}'}

def pairwise_comparisons(df: pd.DataFrame, correction: str = 'holm') -> pd.DataFrame:
    """
    Perform pairwise t-tests with multiple comparison correction.
    
    Args:
        df: DataFrame with experiment data
        correction: Multiple comparison correction method ('holm', 'bonferroni', etc.)
        
    Returns:
        DataFrame with pairwise comparison results
    """
    from statsmodels.stats.multitest import multipletests
    
    conditions = ['virtue', 'neutral', 'vice']
    comparisons = []
    t_stats = []
    p_values = []
    mean_diffs = []
    cohens_d = []
    
    # Get all pairs
    pairs = [(conditions[i], conditions[j]) for i in range(len(conditions)) 
             for j in range(i+1, len(conditions))]
    
    for cond1, cond2 in pairs:
        # Get data for each condition
        data1 = df[df['condition'] == cond1]['compression_ratio'].values
        data2 = df[df['condition'] == cond2]['compression_ratio'].values
        
        if len(data1) > 1 and len(data2) > 1:
            # Paired t-test if same participants, independent otherwise
            participants1 = set(df[df['condition'] == cond1]['participant_id'])
            participants2 = set(df[df['condition'] == cond2]['participant_id'])
            
            if participants1 == participants2:
                # Paired t-test
                t_stat, p_val = stats.ttest_rel(data1, data2)
                # Cohen's d for paired samples
                d = np.mean(data1 - data2) / np.std(data1 - data2)
            else:
                # Independent t-test
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                # Cohen's d for independent samples
                pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
                d = (np.mean(data1) - np.mean(data2)) / pooled_std
        else:
            t_stat, p_val, d = np.nan, np.nan, np.nan
        
        comparisons.append(f"{cond1} vs {cond2}")
        t_stats.append(t_stat)
        p_values.append(p_val)
        mean_diffs.append(np.mean(data1) - np.mean(data2) if len(data1) > 0 and len(data2) > 0 else np.nan)
        cohens_d.append(d)
    
    # Apply multiple comparison correction
    if any(not np.isnan(p) for p in p_values):
        valid_indices = [i for i, p in enumerate(p_values) if not np.isnan(p)]
        valid_pvals = [p_values[i] for i in valid_indices]
        
        _, adj_pvals, _, _ = multipletests(valid_pvals, alpha=0.05, method=correction)
        
        # Reconstruct full list with corrected p-values
        adj_pvals_full = [np.nan] * len(p_values)
        for idx, adj_idx in enumerate(valid_indices):
            adj_pvals_full[adj_idx] = adj_pvals[idx]
    else:
        adj_pvals_full = p_values
    
    # Create results DataFrame
    results = pd.DataFrame({
        'comparison': comparisons,
        't_statistic': [round(t, 4) if not np.isnan(t) else t for t in t_stats],
        'p_value': [round(p, 6) if not np.isnan(p) else p for p in p_values],
        'p_value_adj': [round(p, 6) if not np.isnan(p) else p for p in adj_pvals_full],
        'mean_difference': [round(d, 4) if not np.isnan(d) else d for d in mean_diffs],
        'cohens_d': [round(d, 4) if not np.isnan(d) else d for d in cohens_d],
        'significant': [p < 0.05 if not np.isnan(p) else False for p in adj_pvals_full]
    })
    
    return results

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create publication-ready visualizations.
    
    Args:
        df: DataFrame with experiment data
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Box plot with individual data points
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='condition', y='compression_ratio', 
                order=['virtue', 'neutral', 'vice'], width=0.5)
    sns.stripplot(data=df, x='condition', y='compression_ratio',
                  order=['virtue', 'neutral', 'vice'], 
                  color='black', alpha=0.5, jitter=True, size=4)
    
    plt.title('Compression Ratio by Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Compression Ratio (C_R)', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add hypothesis pattern annotation
    plt.text(0.5, 0.95, 'Predicted: virtue < neutral < vice', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             ha='center', va='top')
    
    # 2. Bar plot with error bars
    ax2 = plt.subplot(1, 2, 2)
    
    stats_df = descriptive_statistics(df)
    conditions = ['virtue', 'neutral', 'vice']
    x_pos = np.arange(len(conditions))
    
    bars = plt.bar(x_pos, [stats_df.loc[c, 'mean'] for c in conditions],
                   yerr=[stats_df.loc[c, 'sem'] for c in conditions],
                   capsize=5, alpha=0.7, color=['#2ecc71', '#3498db', '#e74c3c'])
    
    # Color bars according to hypothesis
    bars[0].set_color('#2ecc71')  # Green for virtue
    bars[1].set_color('#3498db')  # Blue for neutral
    bars[2].set_color('#e74c3c')  # Red for vice
    
    plt.xticks(x_pos, conditions, fontsize=11)
    plt.title('Mean Compression Ratio (± SEM)', fontsize=14, fontweight='bold')
    plt.ylabel('Compression Ratio (C_R)', fontsize=12)
    plt.xlabel('Condition', fontsize=12)
    
    # Add values on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, [stats_df.loc[c, 'mean'] for c in conditions])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'compression_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. Individual participant trajectories (if repeated measures)
    participant_counts = df['participant_id'].value_counts()
    complete_participants = participant_counts[participant_counts == 3].index
    
    if len(complete_participants) >= 3:  # Need at least 3 for a sensible plot
        plt.figure(figsize=(12, 6))
        
        complete_df = df[df['participant_id'].isin(complete_participants)]
        
        # Plot individual trajectories
        for pid in complete_participants[:10]:  # Limit to first 10 for clarity
            participant_data = complete_df[complete_df['participant_id'] == pid]
            plt.plot(['virtue', 'neutral', 'vice'], 
                     participant_data.sort_values('condition')['compression_ratio'].values,
                     'o-', alpha=0.5, linewidth=1, markersize=4)
        
        # Plot group mean trajectory
        group_means = complete_df.groupby('condition')['compression_ratio'].mean()
        plt.plot(['virtue', 'neutral', 'vice'], group_means.values,
                 'k-', linewidth=3, marker='s', markersize=10, 
                 label='Group Mean', zorder=10)
        
        plt.title('Individual Compression Trajectories', fontsize=14, fontweight='bold')
        plt.xlabel('Condition', fontsize=12)
        plt.ylabel('Compression Ratio (C_R)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'individual_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(df: pd.DataFrame, output_dir: str) -> Dict[str, any]:
    """
    Generate complete analysis report.
    
    Args:
        df: DataFrame with experiment data
        output_dir: Directory to save report
        
    Returns:
        Dictionary with all analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("VIRTUOUS SPEECH COMPRESSION ANALYSIS REPORT")
    print("="*70)
    
    # 1. Sample information
    print(f"\n1. SAMPLE INFORMATION")
    print("-"*40)
    print(f"Total observations: {len(df)}")
    print(f"Unique participants: {df['participant_id'].nunique()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    
    # 2. Descriptive statistics
    print(f"\n2. DESCRIPTIVE STATISTICS")
    print("-"*40)
    desc_stats = descriptive_statistics(df)
    print(desc_stats.to_string())
    
    # Save descriptive stats
    desc_stats.to_csv(output_path / 'descriptive_statistics.csv')
    
    # 3. Hypothesis pattern check
    print(f"\n3. HYPOTHESIS PATTERN CHECK")
    print("-"*40)
    hypothesis_check = check_hypothesis_pattern(df)
    
    print(f"Mean C_R(virtue): {hypothesis_check['virtue_mean']:.4f}")
    print(f"Mean C_R(neutral): {hypothesis_check['neutral_mean']:.4f}")
    print(f"Mean C_R(vice): {hypothesis_check['vice_mean']:.4f}")
    print(f"\nPattern check: virtue < neutral < vice")
    print(f"  virtue < neutral: {hypothesis_check['pattern_virtue_lt_neutral']}")
    print(f"  neutral < vice: {hypothesis_check['pattern_neutral_lt_vice']}")
    print(f"  Full pattern: {hypothesis_check['pattern_full']}")
    
    if hypothesis_check['pattern_full']:
        print("✓ PRIMARY HYPOTHESIS SUPPORTED")
        print(f"  C_R decreases as predicted: {hypothesis_check['virtue_mean']:.3f} < {hypothesis_check['neutral_mean']:.3f} < {hypothesis_check['vice_mean']:.3f}")
    else:
        print("✗ PRIMARY HYPOTHESIS NOT SUPPORTED")
        if not hypothesis_check['pattern_virtue_lt_neutral']:
            print(f"  Failed: virtue ({hypothesis_check['virtue_mean']:.3f}) not < neutral ({hypothesis_check['neutral_mean']:.3f})")
        if not hypothesis_check['pattern_neutral_lt_vice']:
            print(f"  Failed: neutral ({hypothesis_check['neutral_mean']:.3f}) not < vice ({hypothesis_check['vice_mean']:.3f})")
    
    # 4. Statistical tests
    print(f"\n4. STATISTICAL TESTS")
    print("-"*40)
    
    # Repeated measures ANOVA
    anova_results = repeated_measures_anova(df)
    if 'error' in anova_results:
        print(f"ANOVA: {anova_results['error']}")
    else:
        print(f"Repeated Measures ANOVA:")
        print(f"  F({anova_results['df_conditions']}, {anova_results['df_error']}) = {anova_results['f_statistic']:.3f}")
        print(f"  p = {anova_results['p_value']:.6f}")
        print(f"  η² = {anova_results['eta_squared']:.3f}")
        print(f"  Complete cases: n = {anova_results['n_complete']}")
        
        if anova_results['p_value'] < 0.05:
            print("  ✓ Significant condition effect")
        else:
            print("  ✗ No significant condition effect")
    
    # Pairwise comparisons
    print(f"\nPairwise Comparisons (Holm-corrected):")
    pairwise_results = pairwise_comparisons(df, correction='holm')
    print(pairwise_results.to_string(index=False))
    
    # Save pairwise results
    pairwise_results.to_csv(output_path / 'pairwise_comparisons.csv')
    
    # 5. Effect size interpretation
    print(f"\n5. EFFECT SIZE INTERPRETATION")
    print("-"*40)
    
    if 'cohens_d' in pairwise_results.columns:
        for _, row in pairwise_results.iterrows():
            if not np.isnan(row['cohens_d']):
                d = abs(row['cohens_d'])
                if d < 0.2:
                    size = 'negligible'
                elif d < 0.5:
                    size = 'small'
                elif d < 0.8:
                    size = 'medium'
                else:
                    size = 'large'
                
                print(f"{row['comparison']}: Cohen's d = {row['cohens_d']:.3f} ({size})")
    
    # 6. Create visualizations
    print(f"\n6. VISUALIZATIONS")
    print("-"*40)
    create_visualizations(df, output_path)
    print(f"Plots saved to: {output_path}/")
    
    # 7. Save complete report
    report_data = {
        'sample_info': {
            'n_observations': len(df),
            'n_participants': df['participant_id'].nunique(),
            'conditions': df['condition'].unique().tolist()
        },
        'descriptive_statistics': desc_stats.to_dict(),
        'hypothesis_check': hypothesis_check,
        'anova_results': anova_results,
        'pairwise_comparisons': pairwise_results.to_dict('records'),
        'analysis_date': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path / 'analysis_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\nComplete report saved to: {output_path / 'analysis_report.json'}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return report_data

def main():
    parser = argparse.ArgumentParser(
        description='Analyze speech compression experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analysis.py --data results.csv --output analysis/
  python analysis.py --data results.csv --output analysis/ --plot-only
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='analysis_results',
                       help='Output directory for analysis results')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plots, skip statistical tests')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data}")
    try:
        df = load_data(args.data)
        print(f"Successfully loaded {len(df)} observations")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run analysis
    if args.plot_only:
        print("\nGenerating plots only...")
        output_path = Path(args.output)
        create_visualizations(df, output_path)
        print(f"Plots saved to: {output_path}")
    else:
        report = generate_report(df, args.output)
        
        # Print summary conclusion
        print("\n" + "="*70)
        print("SUMMARY CONCLUSION")
        print("="*70)
        
        hypothesis_check = report['hypothesis_check']
        if hypothesis_check.get('pattern_full', False):
            print("✓ SUPPORT FOR PREDICTION B1")
            print("  Virtuous speech compresses more than vicious speech.")
            print(f"  Pattern: {hypothesis_check['virtue_mean']:.3f} < {hypothesis_check['neutral_mean']:.3f} < {hypothesis_check['vice_mean']:.3f}")
        else:
            print("✗ NO SUPPORT FOR PREDICTION B1")
            print("  Virtuous speech does not compress more than vicious speech.")
        
        anova_results = report['anova_results']
        if 'p_value' in anova_results and anova_results['p_value'] < 0.05:
            print("✓ Statistically significant condition effect found")
        elif 'p_value' in anova_results:
            print("✗ No statistically significant condition effect")
        
        print("\nSee detailed results in output directory.")

if __name__ == '__main__':
    main()
