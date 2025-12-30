#!/usr/bin/env python3
"""
Complete HRV analysis for Predictions B2 & M1.
Combines HRV processing, crystal study, and statistical analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import processing modules
try:
    from process_hrv import batch_process_hrv, generate_summary
    from crystal_study import CrystalStudyAnalyzer
except ImportError:
    print("Warning: Could not import process_hrv or crystal_study modules")
    print("Make sure they are in the same directory")

class HRVAnalysis:
    """Complete HRV analysis pipeline."""
    
    def __init__(self):
        self.primary_metrics = ['sampen', 'coherence_score', 'lf_hf_ratio']
        
    def run_complete_analysis(self, data_dir: str, output_dir: str, 
                             study_design: str = 'within') -> Dict[str, any]:
        """
        Run complete HRV analysis pipeline.
        
        Args:
            data_dir: Directory containing raw HRV data
            output_dir: Directory to save results
            study_design: 'within' (within-subjects) or 'between' (between-subjects)
            
        Returns:
            Dictionary with all analysis results
        """
        print("="*70)
        print("COMPLETE HRV ANALYSIS PIPELINE")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        results = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'study_design': study_design,
            'data_directory': data_dir,
            'output_directory': str(output_path)
        }
        
        # Step 1: Process HRV data
        print("\n1. PROCESSING HRV DATA")
        print("-"*40)
        
        try:
            hrv_results = batch_process_hrv(data_dir, str(output_path / 'processed'))
            
            if len(hrv_results) == 0:
                print("✗ No HRV data processed successfully")
                return results
            
            results['hrv_processing'] = {
                'n_files_processed': len(hrv_results),
                'n_participants': hrv_results['participant_id'].nunique(),
                'conditions': hrv_results['condition'].unique().tolist()
            }
            
            # Save processed data
            hrv_results.to_csv(output_path / 'hrv_processed_data.csv', index=False)
            print(f"✓ Processed {len(hrv_results)} HRV recordings")
            
        except Exception as e:
            print(f"✗ HRV processing failed: {e}")
            results['hrv_processing'] = {'error': str(e)}
            return results
        
        # Step 2: Analyze Prediction B2 (Moral states affect HRV entropy)
        print("\n2. ANALYZING PREDICTION B2: MORAL STATES & HRV ENTROPY")
        print("-"*40)
        
        b2_results = self.analyze_prediction_b2(hrv_results, output_path)
        results['prediction_b2'] = b2_results
        
        # Step 3: If crystal data exists, analyze Prediction M1
        print("\n3. ANALYZING PREDICTION M1: CRYSTAL AMPLIFICATION")
        print("-"*40)
        
        m1_results = self.analyze_prediction_m1(hrv_results, output_path)
        results['prediction_m1'] = m1_results
        
        # Step 4: Generate comprehensive report
        print("\n4. GENERATING COMPREHENSIVE REPORT")
        print("-"*40)
        
        comprehensive_report = self.generate_comprehensive_report(results, output_path)
        results['comprehensive_report'] = comprehensive_report
        
        # Save complete results
        results_file = output_path / 'hrv_analysis_complete_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Complete analysis saved to: {results_file}")
        
        # Print final conclusions
        self.print_conclusions(results)
        
        return results
    
    def analyze_prediction_b2(self, hrv_data: pd.DataFrame, 
                            output_dir: Path) -> Dict[str, any]:
        """
        Analyze Prediction B2: Virtuous states reduce HRV entropy.
        
        Args:
            hrv_data: Processed HRV data
            output_dir: Output directory
            
        Returns:
            Analysis results for Prediction B2
        """
        results = {}
        
        # Check if we have the required conditions
        required_conditions = ['virtue', 'neutral', 'vice']
        available_conditions = [c for c in required_conditions if c in hrv_data['condition'].unique()]
        
        if len(available_conditions) < 2:
            print(f"✗ Insufficient conditions for B2 analysis")
            print(f"  Required: {required_conditions}")
            print(f"  Available: {hrv_data['condition'].unique()}")
            return {'error': 'Insufficient conditions', 'available': list(hrv_data['condition'].unique())}
        
        print(f"Analyzing conditions: {available_conditions}")
        
        # Create analysis directory
        b2_dir = output_dir / 'prediction_b2'
        b2_dir.mkdir(exist_ok=True, parents=True)
        
        # Calculate descriptive statistics
        desc_stats = {}
        for condition in available_conditions:
            condition_data = hrv_data[hrv_data['condition'] == condition]
            
            desc_stats[condition] = {
                'n': len(condition_data),
                'mean_sampen': float(condition_data['sampen'].mean()),
                'std_sampen': float(condition_data['sampen'].std()),
                'mean_coherence': float(condition_data['coherence_score'].mean()),
                'std_coherence': float(condition_data['coherence_score'].std()),
                'mean_lf_hf': float(condition_data['lf_hf_ratio'].mean()),
                'std_lf_hf': float(condition_data['lf_hf_ratio'].std())
            }
        
        results['descriptive_statistics'] = desc_stats
        
        # Check hypothesis pattern: SampEn(virtue) < SampEn(neutral) < SampEn(vice)
        if 'virtue' in available_conditions and 'neutral' in available_conditions and 'vice' in available_conditions:
            virtue_mean = desc_stats['virtue']['mean_sampen']
            neutral_mean = desc_stats['neutral']['mean_sampen']
            vice_mean = desc_stats['vice']['mean_sampen']
            
            pattern_virtue_lt_neutral = virtue_mean < neutral_mean
            pattern_neutral_lt_vice = neutral_mean < vice_mean
            pattern_full = pattern_virtue_lt_neutral and pattern_neutral_lt_vice
            
            results['hypothesis_check'] = {
                'sampen_virtue': virtue_mean,
                'sampen_neutral': neutral_mean,
                'sampen_vice': vice_mean,
                'pattern_virtue_lt_neutral': pattern_virtue_lt_neutral,
                'pattern_neutral_lt_vice': pattern_neutral_lt_vice,
                'pattern_full': pattern_full,
                'hypothesis_supported': pattern_full
            }
            
            print(f"\nSample Entropy by Condition:")
            print(f"  Virtue:   {virtue_mean:.4f}")
            print(f"  Neutral:  {neutral_mean:.4f}")
            print(f"  Vice:     {vice_mean:.4f}")
            print(f"\nPattern check: virtue < neutral < vice")
            print(f"  virtue < neutral: {'✓' if pattern_virtue_lt_neutral else '✗'}")
            print(f"  neutral < vice:   {'✓' if pattern_neutral_lt_vice else '✗'}")
            print(f"  Full pattern:     {'✓' if pattern_full else '✗'}")
        
        # Statistical tests
        statistical_tests = {}
        
        # ANOVA for condition effect on SampEn
        if len(available_conditions) >= 2:
            try:
                # Prepare data for ANOVA
                groups = [hrv_data[hrv_data['condition'] == c]['sampen'].values 
                         for c in available_conditions]
                
                # One-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                
                statistical_tests['anova_sampen'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'n_groups': len(available_conditions),
                    'groups': available_conditions
                }
                
                print(f"\nANOVA for SampEn:")
                print(f"  F({len(available_conditions)-1}, {len(hrv_data)-len(available_conditions)}) = {f_stat:.3f}")
                print(f"  p = {p_value:.4f}")
                print(f"  Significant: {'✓' if p_value < 0.05 else '✗'}")
                
            except Exception as e:
                statistical_tests['anova_sampen'] = {'error': str(e)}
                print(f"✗ ANOVA failed: {e}")
        
        # Pairwise comparisons
        pairwise_results = []
        if len(available_conditions) >= 2:
            for i in range(len(available_conditions)):
                for j in range(i+1, len(available_conditions)):
                    cond1 = available_conditions[i]
                    cond2 = available_conditions[j]
                    
                    data1 = hrv_data[hrv_data['condition'] == cond1]['sampen'].values
                    data2 = hrv_data[hrv_data['condition'] == cond2]['sampen'].values
                    
                    if len(data1) > 1 and len(data2) > 1:
                        # Check if same participants (paired) or different (independent)
                        participants1 = set(hrv_data[hrv_data['condition'] == cond1]['participant_id'])
                        participants2 = set(hrv_data[hrv_data['condition'] == cond2]['participant_id'])
                        
                        if participants1 == participants2:
                            # Paired t-test
                            t_stat, p_val = stats.ttest_rel(data1, data2)
                            test_type = 'paired'
                        else:
                            # Independent t-test
                            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                            test_type = 'independent'
                        
                        # Effect size
                        mean_diff = np.mean(data1) - np.mean(data2)
                        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
                        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
                        
                        pairwise_results.append({
                            'comparison': f"{cond1} vs {cond2}",
                            't_statistic': float(t_stat),
                            'p_value': float(p_val),
                            'mean_difference': float(mean_diff),
                            'cohens_d': float(cohens_d),
                            'test_type': test_type,
                            'significant': p_val < 0.05
                        })
        
        statistical_tests['pairwise_comparisons'] = pairwise_results
        results['statistical_tests'] = statistical_tests
        
        # Create visualizations
        self.create_b2_visualizations(hrv_data, available_conditions, b2_dir)
        
        # Save B2 results
        b2_results_file = b2_dir / 'prediction_b2_results.json'
        with open(b2_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ B2 analysis saved to: {b2_results_file}")
        
        return results
    
    def analyze_prediction_m1(self, hrv_data: pd.DataFrame, 
                            output_dir: Path) -> Dict[str, any]:
        """
        Analyze Prediction M1: Crystals amplify HRV coherence.
        
        Args:
            hrv_data: Processed HRV data
            output_dir: Output directory
            
        Returns:
            Analysis results for Prediction M1
        """
        results = {}
        
        # Check if we have crystal/placebo data
        # Look for specific conditions or object_type column
        if 'object_type' in hrv_data.columns:
            crystal_data = hrv_data[hrv_data['object_type'] == 'crystal']
            placebo_data = hrv_data[hrv_data['object_type'] == 'placebo']
        else:
            # Try to infer from condition names
            crystal_conditions = [c for c in hrv_data['condition'].unique() 
                                 if any(keyword in c.lower() 
                                        for keyword in ['crystal', 'quartz', 'mineral'])]
            placebo_conditions = [c for c in hrv_data['condition'].unique() 
                                 if any(keyword in c.lower() 
                                        for keyword in ['placebo', 'glass', 'control'])]
            
            if not crystal_conditions or not placebo_conditions:
                print("✗ No crystal/placebo data found for M1 analysis")
                return {'error': 'No crystal/placebo data found'}
            
            crystal_data = hrv_data[hrv_data['condition'].isin(crystal_conditions)]
            placebo_data = hrv_data[hrv_data['condition'].isin(placebo_conditions)]
        
        if len(crystal_data) == 0 or len(placebo_data) == 0:
            print("✗ Insufficient crystal/placebo data for M1 analysis")
            return {'error': 'Insufficient crystal/placebo data'}
        
        print(f"Crystal data: {len(crystal_data)} observations")
        print(f"Placebo data: {len(placebo_data)} observations")
        
        # Create analysis directory
        m1_dir = output_dir / 'prediction_m1'
        m1_dir.mkdir(exist_ok=True, parents=True)
        
        # Save separate CSV files for crystal study analysis
        crystal_file = m1_dir / 'crystal_data.csv'
        placebo_file = m1_dir / 'placebo_data.csv'
        
        crystal_data.to_csv(crystal_file, index=False)
        placebo_data.to_csv(placebo_file, index=False)
        
        print(f"✓ Saved crystal data to: {crystal_file}")
        print(f"✓ Saved placebo data to: {placebo_file}")
        
        # Run crystal study analysis if module available
        try:
            analyzer = CrystalStudyAnalyzer()
            
            # Load data
            crystal_df, placebo_df = analyzer.load_study_data(str(crystal_file), str(placebo_file))
            
            # Match participants
            combined_df, change_df = analyzer.match_participants(crystal_df, placebo_df)
            
            # Analyze crystal effect
            crystal_results = analyzer.analyze_crystal_effect(change_df)
            
            # Create visualizations
            analyzer.create_visualizations(combined_df, change_df, crystal_results, m1_dir)
            
            # Generate report
            report = analyzer.generate_report(crystal_results, m1_dir)
            
            results = {
                'crystal_results': crystal_results,
                'report': report,
                'files': {
                    'crystal_data': str(crystal_file),
                    'placebo_data': str(placebo_file),
                    'combined_data': str(m1_dir / 'combined_data.csv'),
                    'change_scores': str(m1_dir / 'change_scores.csv')
                }
            }
            
            # Save combined data
            combined_df.to_csv(m1_dir / 'combined_data.csv', index=False)
            change_df.to_csv(m1_dir / 'change_scores.csv', index=False)
            
            print(f"\n✓ M1 analysis complete")
            
        except Exception as e:
            print(f"✗ Crystal study analysis failed: {e}")
            results = {'error': str(e)}
        
        return results
    
    def create_b2_visualizations(self, hrv_data: pd.DataFrame, 
                               conditions: List[str], output_dir: Path):
        """Create visualizations for Prediction B2."""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Sample Entropy by Condition
        plt.figure(figsize=(10, 6))
        
        # Box plot
        ax1 = plt.subplot(1, 2, 1)
        sns.boxplot(data=hrv_data, x='condition', y='sampen', 
                   order=conditions, ax=ax1, width=0.5)
        sns.stripplot(data=hrv_data, x='condition', y='sampen',
                     order=conditions, ax=ax1, color='black', 
                     alpha=0.5, jitter=True, size=4)
        
        ax1.set_title('Sample Entropy by Condition', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Condition', fontsize=12)
        ax1.set_ylabel('Sample Entropy (lower = more coherent)', fontsize=11)
        
        # Add hypothesis pattern
        if len(conditions) == 3 and 'virtue' in conditions and 'vice' in conditions:
            ax1.text(0.5, 0.95, 'Predicted: virtue < neutral < vice', 
                    transform=ax1.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    ha='center', va='top')
        
        # 2. Coherence Score by Condition
        ax2 = plt.subplot(1, 2, 2)
        
        # Calculate means and SEM
        means = hrv_data.groupby('condition')['coherence_score'].mean()
        sems = hrv_data.groupby('condition')['coherence_score'].sem()
        
        x_pos = np.arange(len(conditions))
        bars = ax2.bar(x_pos, [means.get(c, 0) for c in conditions],
                      yerr=[sems.get(c, 0) for c in conditions],
                      capsize=5, alpha=0.7)
        
        # Color by condition type
        for i, condition in enumerate(conditions):
            if condition == 'virtue':
                bars[i].set_color('#2ecc71')  # Green
            elif condition == 'vice':
                bars[i].set_color('#e74c3c')  # Red
            else:
                bars[i].set_color('#3498db')  # Blue
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(conditions, fontsize=11)
        ax2.set_title('HRV Coherence by Condition', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Coherence Score (0-100)', fontsize=11)
        ax2.set_xlabel('Condition', fontsize=12)
        
        # Add values on bars
        for i, (bar, mean_val) in enumerate(zip(bars, [means.get(c, 0) for c in conditions])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Prediction B2: Moral States Affect HRV Entropy', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'b2_sample_entropy_coherence.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plot: SampEn vs Coherence
        plt.figure(figsize=(8, 6))
        
        # Color by condition
        condition_colors = {
            'virtue': '#2ecc71',
            'neutral': '#3498db',
            'vice': '#e74c3c'
        }
        
        for condition in conditions:
            condition_data = hrv_data[hrv_data['condition'] == condition]
            color = condition_colors.get(condition, '#95a5a6')
            
            plt.scatter(condition_data['sampen'], condition_data['coherence_score'],
                       color=color, alpha=0.6, s=50, label=condition)
        
        plt.xlabel('Sample Entropy', fontsize=12)
        plt.ylabel('Coherence Score', fontsize=12)
        plt.title('Relationship: Entropy vs. Coherence', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation
        if len(hrv_data) > 2:
            corr_coef = np.corrcoef(hrv_data['sampen'], hrv_data['coherence_score'])[0, 1]
            plt.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=plt.gca().transAxes,
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'b2_entropy_vs_coherence.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ B2 visualizations saved to: {output_dir}")
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, any], 
                                    output_dir: Path) -> Dict[str, any]:
        """Generate comprehensive analysis report."""
        
        report = {
            'summary': {},
            'conclusions': {},
            'recommendations': []
        }
        
        # Check B2 results
        b2_results = analysis_results.get('prediction_b2', {})
        b2_hypothesis = b2_results.get('hypothesis_check', {})
        
        b2_supported = b2_hypothesis.get('hypothesis_supported', False)
        b2_pattern_full = b2_hypothesis.get('pattern_full', False)
        
        report['summary']['prediction_b2'] = {
            'hypothesis_supported': b2_supported,
            'pattern_full': b2_pattern_full,
            'sampen_virtue': b2_hypothesis.get('sampen_virtue', 'N/A'),
            'sampen_vice': b2_hypothesis.get('sampen_vice', 'N/A')
        }
        
        # Check M1 results
        m1_results = analysis_results.get('prediction_m1', {})
        m1_report = m1_results.get('report', {})
        m1_summary = m1_report.get('summary', {})
        
        m1_supported = m1_summary.get('primary_hypothesis_supported', False)
        
        report['summary']['prediction_m1'] = {
            'hypothesis_supported': m1_supported,
            'n_significant': m1_summary.get('n_significant', 0),
            'n_hypothesis_supported': m1_summary.get('n_hypothesis_supported', 0)
        }
        
        # Generate conclusions
        conclusions = []
        
        if b2_supported:
            conclusions.append("✓ Prediction B2 SUPPORTED: Virtuous states reduce HRV entropy.")
            conclusions.append(f"  Pattern: SampEn(virtue) < SampEn(neutral) < SampEn(vice)")
        else:
            conclusions.append("✗ Prediction B2 NOT SUPPORTED: Virtuous states do not reduce HRV entropy.")
        
        if m1_supported:
            conclusions.append("✓ Prediction M1 SUPPORTED: Quartz crystals amplify HRV coherence.")
        else:
            conclusions.append("✗ Prediction M1 NOT SUPPORTED: Quartz crystals do not amplify HRV coherence.")
        
        report['conclusions']['text'] = conclusions
        
        # Generate recommendations
        recommendations = []
        
        if not b2_supported:
            recommendations.append("1. Increase sample size for B2 analysis")
            recommendations.append("2. Standardize meditation protocols across conditions")
            recommendations.append("3. Control for time of day and participant state")
        
        if not m1_supported:
            recommendations.append("1. Ensure proper blinding in crystal/placebo study")
            recommendations.append("2. Test different crystal types and sizes")
            recommendations.append("3. Measure participant belief about crystal efficacy")
        
        if b2_supported and m1_supported:
            recommendations.append("1. Conduct combined study with crystal during virtuous meditation")
            recommendations.append("2. Test dose-response relationship with crystal exposure")
            recommendations.append("3. Investigate neural correlates using fMRI/EEG")
        
        report['recommendations'] = recommendations
        
        # Save report
        report_file = output_dir / 'comprehensive_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Comprehensive report saved to: {report_file}")
        
        return report
    
    def print_conclusions(self, analysis_results: Dict[str, any]):
        """Print final conclusions."""
        
        print("\n" + "="*70)
        print("FINAL CONCLUSIONS")
        print("="*70)
        
        # B2 Conclusion
        b2_results = analysis_results.get('prediction_b2', {})
        b2_hypothesis = b2_results.get('hypothesis_check', {})
        b2_supported = b2_hypothesis.get('hypothesis_supported', False)
        
        if b2_supported:
            print("\n✓ PREDICTION B2: SUPPORTED")
            print("  Virtuous mental states reduce HRV entropy (increase coherence).")
            print("  This supports the moral thermodynamics hypothesis.")
        else:
            print("\n✗ PREDICTION B2: NOT SUPPORTED")
            print("  Virtuous states do not reliably reduce HRV entropy.")
        
        # M1 Conclusion
        m1_results = analysis_results.get('prediction_m1', {})
        m1_report = m1_results.get('report', {})
        m1_summary = m1_report.get('summary', {})
        m1_supported = m1_summary.get('primary_hypothesis_supported', False)
        
        if m1_supported:
            print("\n✓ PREDICTION M1: SUPPORTED")
            print("  Quartz crystals amplify the HRV coherence effect.")
            print("  This supports the crystalline virtue hypothesis.")
        else:
            print("\n✗ PREDICTION M1: NOT SUPPORTED")
            print("  Crystals do not amplify HRV coherence beyond placebo.")
        
        # Overall conclusion
        if b2_supported and m1_supported:
            print("\n" + "="*70)
            print("DOUBLE CONFIRMATION: Both hypotheses supported")
            print("This provides strong evidence for:")
            print("1. Moral thermodynamics (virtue reduces entropy)")
            print("2. Material amplification (crystals enhance coherence)")
            print("="*70)
        elif b2_supported or m1_supported:
            print("\n" + "="*70)
            print("PARTIAL CONFIRMATION: One hypothesis supported")
            print("Further research needed to confirm the full model.")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("NO SUPPORT: Neither hypothesis supported")
            print("The model requires revision or different experimental approach.")
            print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Complete HRV analysis for Predictions B2 & M1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Complete analysis pipeline
  python hrv_analysis.py --data_dir ./raw_hrv_data/ --output ./analysis_results/
  
  # Just analyze Prediction B2
  python hrv_analysis.py --data_dir ./data/ --output ./results/ --only b2
  
  # Use existing processed data
  python hrv_analysis.py --processed_data ./processed/hrv_processed_data.csv --output ./results/
        """
    )
    
    parser.add_argument('--data_dir', type=str, help='Directory with raw HRV data files')
    parser.add_argument('--processed_data', type=str, help='Path to already processed HRV data CSV')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--only', choices=['b2', 'm1', 'both'], default='both',
                       help='Which prediction to analyze')
    parser.add_argument('--design', choices=['within', 'between'], default='within',
                       help='Study design (within-subjects or between-subjects)')
    
    args = parser.parse_args()
    
    if not args.data_dir and not args.processed_data:
        print("Error: Must provide either --data_dir or --processed_data")
        parser.print_help()
        return
    
    # Create analyzer
    analyzer = HRVAnalysis()
    
    if args.processed_data:
        # Use already processed data
        print(f"Loading processed data from: {args.processed_data}")
        try:
            hrv_data = pd.read_csv(args.processed_data)
            print(f"Loaded {len(hrv_data)} observations")
            
            # Run analysis based on which prediction requested
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            if args.only in ['b2', 'both']:
                print("\nAnalyzing Prediction B2...")
                b2_results = analyzer.analyze_prediction_b2(hrv_data, output_dir)
            
            if args.only in ['m1', 'both']:
                print("\nAnalyzing Prediction M1...")
                m1_results = analyzer.analyze_prediction_m1(hrv_data, output_dir)
            
        except Exception as e:
            print(f"Error: {e}")
            return
    
    else:
        # Run complete pipeline
        print(f"Starting complete analysis pipeline...")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output}")
        print(f"Analyzing: {args.only}")
        print(f"Study design: {args.design}")
        
        results = analyzer.run_complete_analysis(
            args.data_dir, 
            args.output, 
            args.design
        )

if __name__ == '__main__':
    main()
