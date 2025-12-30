#!/usr/bin/env python3
"""
Crystal Amplification Study (Prediction M1)
Analysis of HRV coherence with quartz crystals vs. placebo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class CrystalStudyAnalyzer:
    """Analyze crystal amplification effects on HRV coherence."""
    
    def __init__(self):
        self.metrics = ['sampen', 'coherence_score', 'lf_hf_ratio', 'sd1_sd2_ratio']
    
    def load_study_data(self, crystal_data: str, placebo_data: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load crystal and placebo study data.
        
        Args:
            crystal_data: Path to crystal condition results
            placebo_data: Path to placebo condition results
            
        Returns:
            Tuple of (crystal_df, placebo_df) DataFrames
        """
        # Load data
        crystal_df = pd.read_csv(crystal_data)
        placebo_df = pd.read_csv(placebo_data)
        
        # Add condition labels
        crystal_df['object_type'] = 'crystal'
        placebo_df['object_type'] = 'placebo'
        
        # Ensure required columns
        required_cols = ['participant_id', 'condition', 'sampen', 'coherence_score']
        for df in [crystal_df, placebo_df]:
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col} in data")
        
        return crystal_df, placebo_df
    
    def match_participants(self, crystal_df: pd.DataFrame, placebo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match participants across conditions (crossover design).
        
        Args:
            crystal_df: Crystal condition data
            placebo_df: Placebo condition data
            
        Returns:
            Combined DataFrame with matched participants
        """
        # Find common participants
        crystal_participants = set(crystal_df['participant_id'].unique())
        placebo_participants = set(placebo_df['participant_id'].unique())
        common_participants = crystal_participants.intersection(placebo_participants)
        
        print(f"Total participants: Crystal={len(crystal_participants)}, Placebo={len(placebo_participants)}")
        print(f"Matched participants: {len(common_participants)}")
        
        # Filter to common participants
        crystal_matched = crystal_df[crystal_df['participant_id'].isin(common_participants)].copy()
        placebo_matched = placebo_df[placebo_df['participant_id'].isin(common_participants)].copy()
        
        # Combine
        combined = pd.concat([crystal_matched, placebo_matched], ignore_index=True)
        
        # Calculate change scores
        change_scores = []
        for participant in common_participants:
            # Get baseline (neutral condition without object)
            crystal_neutral = crystal_matched[
                (crystal_matched['participant_id'] == participant) & 
                (crystal_matched['condition'] == 'neutral')
            ]
            placebo_neutral = placebo_matched[
                (placebo_matched['participant_id'] == participant) & 
                (placebo_matched['condition'] == 'neutral')
            ]
            
            # Get meditation with object
            crystal_med = crystal_matched[
                (crystal_matched['participant_id'] == participant) & 
                (crystal_matched['condition'] == 'meditation')
            ]
            placebo_med = placebo_matched[
                (placebo_matched['participant_id'] == participant) & 
                (placebo_matched['condition'] == 'meditation')
            ]
            
            if len(crystal_neutral) > 0 and len(crystal_med) > 0:
                # Calculate change from neutral to meditation for crystal
                for metric in self.metrics:
                    if metric in crystal_neutral.columns and metric in crystal_med.columns:
                        neutral_val = crystal_neutral[metric].iloc[0]
                        med_val = crystal_med[metric].iloc[0]
                        change = med_val - neutral_val
                        
                        change_scores.append({
                            'participant_id': participant,
                            'object_type': 'crystal',
                            'metric': metric,
                            'neutral_value': neutral_val,
                            'meditation_value': med_val,
                            'change_score': change,
                            'percent_change': (change / neutral_val * 100) if neutral_val != 0 else 0
                        })
            
            if len(placebo_neutral) > 0 and len(placebo_med) > 0:
                # Calculate change from neutral to meditation for placebo
                for metric in self.metrics:
                    if metric in placebo_neutral.columns and metric in placebo_med.columns:
                        neutral_val = placebo_neutral[metric].iloc[0]
                        med_val = placebo_med[metric].iloc[0]
                        change = med_val - neutral_val
                        
                        change_scores.append({
                            'participant_id': participant,
                            'object_type': 'placebo',
                            'metric': metric,
                            'neutral_value': neutral_val,
                            'meditation_value': med_val,
                            'change_score': change,
                            'percent_change': (change / neutral_val * 100) if neutral_val != 0 else 0
                        })
        
        change_df = pd.DataFrame(change_scores)
        return combined, change_df
    
    def analyze_crystal_effect(self, change_df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze crystal vs. placebo effects.
        
        Args:
            change_df: DataFrame with change scores
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        for metric in self.metrics:
            metric_data = change_df[change_df['metric'] == metric]
            
            if len(metric_data) == 0:
                continue
            
            # Separate crystal and placebo
            crystal_changes = metric_data[metric_data['object_type'] == 'crystal']['change_score'].values
            placebo_changes = metric_data[metric_data['object_type'] == 'placebo']['change_score'].values
            
            if len(crystal_changes) > 1 and len(placebo_changes) > 1:
                # Descriptive statistics
                crystal_mean = np.mean(crystal_changes)
                crystal_std = np.std(crystal_changes)
                placebo_mean = np.mean(placebo_changes)
                placebo_std = np.std(placebo_changes)
                
                # Paired t-test (since same participants)
                # Match participants for paired test
                paired_data = []
                for participant in metric_data['participant_id'].unique():
                    participant_data = metric_data[metric_data['participant_id'] == participant]
                    if len(participant_data) == 2:  # Should have both crystal and placebo
                        crystal_val = participant_data[participant_data['object_type'] == 'crystal']['change_score'].iloc[0]
                        placebo_val = participant_data[participant_data['object_type'] == 'placebo']['change_score'].iloc[0]
                        paired_data.append((crystal_val, placebo_val))
                
                if len(paired_data) >= 2:
                    paired_array = np.array(paired_data)
                    crystal_paired = paired_array[:, 0]
                    placebo_paired = paired_array[:, 1]
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(crystal_paired, placebo_paired)
                    
                    # Effect size (Cohen's d for paired samples)
                    mean_diff = np.mean(crystal_paired - placebo_paired)
                    std_diff = np.std(crystal_paired - placebo_paired)
                    cohens_d = mean_diff / std_diff if std_diff != 0 else 0
                    
                    # Check hypothesis: crystal should reduce SampEn more than placebo
                    hypothesis_supported = False
                    if metric == 'sampen':
                        # Lower SampEn is better (more coherent)
                        hypothesis_supported = crystal_mean < placebo_mean
                    elif metric == 'coherence_score':
                        # Higher coherence is better
                        hypothesis_supported = crystal_mean > placebo_mean
                    elif metric == 'lf_hf_ratio':
                        # Balanced ratio is better (closer to 1.5-2.0)
                        crystal_abs_diff = abs(crystal_mean - 1.75)
                        placebo_abs_diff = abs(placebo_mean - 1.75)
                        hypothesis_supported = crystal_abs_diff < placebo_abs_diff
                    
                    results[metric] = {
                        'crystal_mean': float(crystal_mean),
                        'crystal_std': float(crystal_std),
                        'placebo_mean': float(placebo_mean),
                        'placebo_std': float(placebo_std),
                        'mean_difference': float(crystal_mean - placebo_mean),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'n_paired': len(paired_data),
                        'hypothesis_supported': hypothesis_supported,
                        'significant': p_value < 0.05
                    }
        
        return results
    
    def create_visualizations(self, combined_df: pd.DataFrame, 
                            change_df: pd.DataFrame, 
                            results: Dict[str, any],
                            output_dir: Path):
        """
        Create visualizations for crystal study.
        
        Args:
            combined_df: Combined crystal and placebo data
            change_df: Change scores DataFrame
            results: Analysis results
            output_dir: Directory to save plots
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Change scores comparison (main result)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.metrics[:4]):  # Plot first 4 metrics
            if metric not in change_df['metric'].unique():
                continue
            
            ax = axes[idx]
            metric_data = change_df[change_df['metric'] == metric]
            
            # Box plot of change scores
            sns.boxplot(data=metric_data, x='object_type', y='change_score', 
                       ax=ax, width=0.5)
            sns.stripplot(data=metric_data, x='object_type', y='change_score',
                         ax=ax, color='black', alpha=0.5, jitter=True, size=4)
            
            # Add significance star if significant
            if metric in results and results[metric]['significant']:
                y_max = metric_data['change_score'].max()
                ax.text(0.5, y_max * 1.1, '*', ha='center', va='bottom', 
                       fontsize=20, fontweight='bold', color='red')
            
            ax.set_title(f'{metric.upper()} Change Scores', fontsize=12, fontweight='bold')
            ax.set_xlabel('Object Type', fontsize=10)
            ax.set_ylabel('Change Score', fontsize=10)
            
            # Add mean difference annotation
            if metric in results:
                diff = results[metric]['mean_difference']
                p_val = results[metric]['p_value']
                ax.text(0.5, 0.05, f'Δ = {diff:.3f}\np = {p_val:.4f}', 
                       transform=ax.transAxes, ha='center', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Crystal vs. Placebo: Change in HRV Metrics', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'crystal_vs_placebo_changes.png', dpi=300, bbox_inches='tight')
        
        # 2. Individual participant trajectories
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Sample entropy trajectories
        sampen_data = change_df[change_df['metric'] == 'sampen']
        if len(sampen_data) > 0:
            # Pivot for plotting
            pivot_df = sampen_data.pivot(index='participant_id', 
                                        columns='object_type', 
                                        values='change_score')
            
            # Plot lines connecting crystal and placebo for each participant
            for idx in range(min(20, len(pivot_df))):  # Limit to 20 for clarity
                participant = pivot_df.index[idx]
                crystal_val = pivot_df.loc[participant, 'crystal']
                placebo_val = pivot_df.loc[participant, 'placebo']
                
                axes[0].plot([0, 1], [placebo_val, crystal_val], 'o-', 
                           alpha=0.5, linewidth=1, markersize=4)
            
            axes[0].set_xticks([0, 1])
            axes[0].set_xticklabels(['Placebo', 'Crystal'])
            axes[0].set_title('Individual SampEn Change Trajectories', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Δ SampEn (lower = better)', fontsize=10)
            axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add group means
            placebo_mean = pivot_df['placebo'].mean()
            crystal_mean = pivot_df['crystal'].mean()
            axes[0].plot([0, 1], [placebo_mean, crystal_mean], 'k-', 
                       linewidth=3, marker='s', markersize=10, label='Group Mean')
            axes[0].legend()
        
        # Coherence score trajectories
        coherence_data = change_df[change_df['metric'] == 'coherence_score']
        if len(coherence_data) > 0:
            pivot_df = coherence_data.pivot(index='participant_id', 
                                           columns='object_type', 
                                           values='change_score')
            
            for idx in range(min(20, len(pivot_df))):
                participant = pivot_df.index[idx]
                crystal_val = pivot_df.loc[participant, 'crystal']
                placebo_val = pivot_df.loc[participant, 'placebo']
                
                axes[1].plot([0, 1], [placebo_val, crystal_val], 'o-', 
                           alpha=0.5, linewidth=1, markersize=4)
            
            axes[1].set_xticks([0, 1])
            axes[1].set_xticklabels(['Placebo', 'Crystal'])
            axes[1].set_title('Individual Coherence Change Trajectories', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Δ Coherence Score (higher = better)', fontsize=10)
            axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add group means
            placebo_mean = pivot_df['placebo'].mean()
            crystal_mean = pivot_df['crystal'].mean()
            axes[1].plot([0, 1], [placebo_mean, crystal_mean], 'k-', 
                       linewidth=3, marker='s', markersize=10, label='Group Mean')
            axes[1].legend()
        
        plt.suptitle('Individual Response Patterns', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(output_dir / 'individual_trajectories.png', dpi=300, bbox_inches='tight')
        
        # 3. Effect size visualization
        if results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = list(results.keys())
            effect_sizes = [results[m]['cohens_d'] for m in metrics]
            p_values = [results[m]['p_value'] for m in metrics]
            
            # Color by significance
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            
            bars = ax.bar(range(len(metrics)), effect_sizes, color=colors, alpha=0.7)
            
            # Add value labels
            for i, (bar, es, p) in enumerate(zip(bars, effect_sizes, p_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05 * np.sign(height),
                       f'd = {es:.2f}\np = {p:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9)
            
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.upper() for m in metrics], rotation=45, ha='right')
            ax.set_title('Effect Sizes: Crystal vs. Placebo (Cohen\'s d)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Cohen\'s d (positive favors crystal)', fontsize=12)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
            ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
            
            # Shade region for hypothesis (crystal better)
            if 'sampen' in results:
                # For SampEn, negative effect size is good (crystal reduces entropy more)
                if results['sampen']['mean_difference'] < 0:
                    ax.axhspan(-2, 0, alpha=0.1, color='green', label='Hypothesis region')
            
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')
        print(f"Visualizations saved to: {output_dir}")
    
    def generate_report(self, results: Dict[str, any], output_dir: Path) -> Dict[str, any]:
        """
        Generate analysis report.
        
        Args:
            results: Analysis results
            output_dir: Output directory
            
        Returns:
            Complete report dictionary
        """
        report = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'n_metrics_analyzed': len(results),
            'results': results,
            'summary': {}
        }
        
        # Generate summary
        significant_metrics = [m for m in results if results[m]['significant']]
        hypothesis_supported = [m for m in results if results[m]['hypothesis_supported']]
        
        report['summary'] = {
            'n_significant': len(significant_metrics),
            'significant_metrics': significant_metrics,
            'n_hypothesis_supported': len(hypothesis_supported),
            'hypothesis_supported_metrics': hypothesis_supported,
            'primary_hypothesis_supported': 'sampen' in hypothesis_supported
        }
        
        # Save report
        report_file = output_dir / 'crystal_study_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {report_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Analyze crystal amplification study')
    
    parser.add_argument('--crystal_data', type=str, required=True,
                       help='Path to crystal condition HRV results CSV')
    parser.add_argument('--placebo_data', type=str, required=True,
                       help='Path to placebo condition HRV results CSV')
    parser.add_argument('--output_dir', type=str, default='crystal_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CrystalStudyAnalyzer()
    
    print("="*70)
    print("CRYSTAL AMPLIFICATION STUDY ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n1. LOADING DATA")
    print("-"*40)
    crystal_df, placebo_df = analyzer.load_study_data(args.crystal_data, args.placebo_data)
    print(f"✓ Crystal data: {len(crystal_df)} observations")
    print(f"✓ Placebo data: {len(placebo_df)} observations")
    
    # Match participants
    print("\n2. MATCHING PARTICIPANTS")
    print("-"*40)
    combined_df, change_df = analyzer.match_participants(crystal_df, placebo_df)
    print(f"✓ Combined data: {len(combined_df)} observations")
    print(f"✓ Change scores: {len(change_df)} calculated")
    
    # Analyze crystal effect
    print("\n3. ANALYZING CRYSTAL EFFECT")
    print("-"*40)
    results = analyzer.analyze_crystal_effect(change_df)
    
    # Print results
    for metric, result in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Crystal mean change: {result['crystal_mean']:.4f} ± {result['crystal_std']:.4f}")
        print(f"  Placebo mean change: {result['placebo_mean']:.4f} ± {result['placebo_std']:.4f}")
        print(f"  Difference: {result['mean_difference']:.4f}")
        print(f"  t({result['n_paired']-1}) = {result['t_statistic']:.3f}, p = {result['p_value']:.4f}")
        print(f"  Cohen's d = {result['cohens_d']:.3f}")
        print(f"  Significant: {'✓' if result['significant'] else '✗'}")
        print(f"  Hypothesis supported: {'✓' if result['hypothesis_supported'] else '✗'}")
    
    # Create visualizations
    print("\n4. CREATING VISUALIZATIONS")
    print("-"*40)
    output_dir = Path(args.output_dir)
    analyzer.create_visualizations(combined_df, change_df, results, output_dir)
    
    # Generate report
    print("\n5. GENERATING REPORT")
    print("-"*40)
    report = analyzer.generate_report(results, output_dir)
    
    # Print conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if report['summary']['primary_hypothesis_supported']:
        print("✓ PRIMARY HYPOTHESIS (M1) SUPPORTED")
        print("  Quartz crystals amplify HRV coherence more than placebo.")
        
        if 'sampen' in results:
            crystal_effect = results['sampen']['mean_difference']
            if crystal_effect < 0:
                print(f"  Crystals reduce SampEn by {-crystal_effect:.3f} more than placebo")
            else:
                print(f"  Crystals increase SampEn by {crystal_effect:.3f} more than placebo")
    else:
        print("✗ PRIMARY HYPOTHESIS (M1) NOT SUPPORTED")
        print("  Quartz crystals do not amplify HRV coherence more than placebo.")
    
    print(f"\nSignificant effects found in {report['summary']['n_significant']}/{len(results)} metrics")
    print(f"Hypothesis supported in {report['summary']['n_hypothesis_supported']}/{len(results)} metrics")
    
    print(f"\nComplete analysis saved to: {output_dir}")

if __name__ == '__main__':
    main()
