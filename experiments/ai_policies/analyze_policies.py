#!/usr/bin/env python3
"""
Analyze policy compressibility for AEP vs reward-maximizing agents.
Tests Prediction A1: AEP-trained policies are more compressible.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import gzip
import pickle
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import the policy network from training script
sys.path.append(str(Path(__file__).parent))
from train_agents import AEPPolicyNetwork

class PolicyCompressionAnalyzer:
    """Analyze compressibility of trained policies."""
    
    def __init__(self):
        self.compression_methods = ['gzip', 'pickle', 'numpy']
    
    def analyze_policy_file(self, policy_path: Path) -> Dict[str, Any]:
        """
        Analyze a single saved policy file.
        
        Args:
            policy_path: Path to saved policy (.pt file)
            
        Returns:
            Dictionary with compression analysis results
        """
        # Load the checkpoint
        checkpoint = torch.load(policy_path, map_location='cpu')
        
        # Extract policy state dict
        policy_state = checkpoint['policy_state_dict']
        
        # Convert to different representations and compress
        results = {
            'filename': policy_path.name,
            'file_size_bytes': policy_path.stat().st_size,
            'compression_analysis': {}
        }
        
        # 1. Raw torch state dict compression
        raw_data = pickle.dumps(policy_state)
        compressed_gzip = gzip.compress(raw_data, compresslevel=9)
        
        results['compression_analysis']['torch_state_dict'] = {
            'original_size': len(raw_data),
            'compressed_size': len(compressed_gzip),
            'compression_ratio': len(compressed_gzip) / len(raw_data),
            'compression_gain': 1 - (len(compressed_gzip) / len(raw_data))
        }
        
        # 2. Extract and analyze weight matrices
        weight_data = {}
        weight_entropy = 0
        
        for name, param in policy_state.items():
            if 'weight' in name:
                # Convert to numpy
                weight_np = param.numpy().flatten()
                weight_data[name] = weight_np
                
                # Calculate entropy of weight distribution
                hist, _ = np.histogram(weight_np, bins=50, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                weight_entropy += entropy
        
        # 3. Analyze weight compressibility
        if weight_data:
            # Concatenate all weights
            all_weights = np.concatenate(list(weight_data.values()))
            
            # Raw weight data
            weight_bytes = all_weights.astype(np.float32).tobytes()
            compressed_weights = gzip.compress(weight_bytes, compresslevel=9)
            
            results['compression_analysis']['weights_only'] = {
                'original_size': len(weight_bytes),
                'compressed_size': len(compressed_weights),
                'compression_ratio': len(compressed_weights) / len(weight_bytes),
                'weight_entropy': weight_entropy,
                'weight_mean': float(np.mean(all_weights)),
                'weight_std': float(np.std(all_weights)),
                'weight_sparsity': float(np.mean(np.abs(all_weights) < 0.01))  # Near-zero weights
            }
        
        # 4. Analyze parameter statistics
        all_params = []
        for param in policy_state.values():
            all_params.append(param.numpy().flatten())
        
        if all_params:
            all_params_np = np.concatenate(all_params)
            
            results['parameter_statistics'] = {
                'total_parameters': len(all_params_np),
                'mean_abs_value': float(np.mean(np.abs(all_params_np))),
                'std_value': float(np.std(all_params_np)),
                'min_value': float(np.min(all_params_np)),
                'max_value': float(np.max(all_params_np)),
                'sparsity_001': float(np.mean(np.abs(all_params_np) < 0.001)),
                'sparsity_01': float(np.mean(np.abs(all_params_np) < 0.01)),
                'sparsity_1': float(np.mean(np.abs(all_params_np) < 0.1))
            }
        
        # 5. Network topology analysis (if we can reconstruct the network)
        try:
            # Try to infer network architecture
            layer_sizes = []
            for name in policy_state.keys():
                if 'weight' in name:
                    layer_sizes.append(policy_state[name].shape[1])
            
            results['architecture_inference'] = {
                'inferred_layers': layer_sizes,
                'total_layers': len([k for k in policy_state.keys() if 'weight' in k])
            }
        except:
            pass
        
        return results
    
    def prune_policy(self, policy_state: Dict, sparsity: float = 0.9) -> Dict:
        """
        Prune a policy by setting smallest weights to zero.
        
        Args:
            policy_state: Policy state dictionary
            sparsity: Target sparsity (fraction of weights to prune)
            
        Returns:
            Pruned policy state dictionary
        """
        pruned_state = policy_state.copy()
        
        for name, param in pruned_state.items():
            if 'weight' in name:
                # Flatten and find threshold
                weights = param.numpy().flatten()
                abs_weights = np.abs(weights)
                threshold = np.percentile(abs_weights, sparsity * 100)
                
                # Create mask
                mask = abs_weights > threshold
                pruned_weights = weights * mask.reshape(weights.shape)
                
                # Update parameter
                pruned_state[name] = torch.from_numpy(pruned_weights.reshape(param.shape))
        
        return pruned_state
    
    def analyze_pruning_effect(self, policy_path: Path, 
                              sparsity_levels: List[float] = None) -> Dict[str, Any]:
        """
        Analyze how pruning affects compressibility.
        
        Args:
            policy_path: Path to policy file
            sparsity_levels: List of sparsity levels to test
            
        Returns:
            Dictionary with pruning analysis results
        """
        if sparsity_levels is None:
            sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        
        # Load policy
        checkpoint = torch.load(policy_path, map_location='cpu')
        policy_state = checkpoint['policy_state_dict']
        
        results = {'sparsity_levels': sparsity_levels, 'pruning_analysis': []}
        
        for sparsity in sparsity_levels:
            # Prune policy
            pruned_state = self.prune_policy(policy_state, sparsity)
            
            # Compress pruned policy
            pruned_data = pickle.dumps(pruned_state)
            compressed = gzip.compress(pruned_data, compresslevel=9)
            
            # Calculate actual sparsity
            total_weights = 0
            zero_weights = 0
            for name, param in pruned_state.items():
                if 'weight' in name:
                    weights = param.numpy().flatten()
                    total_weights += len(weights)
                    zero_weights += np.sum(weights == 0)
            
            actual_sparsity = zero_weights / total_weights if total_weights > 0 else 0
            
            analysis = {
                'target_sparsity': sparsity,
                'actual_sparsity': actual_sparsity,
                'original_size': len(pruned_data),
                'compressed_size': len(compressed),
                'compression_ratio': len(compressed) / len(pruned_data),
                'compression_gain': 1 - (len(compressed) / len(pruned_data)),
                'zero_weights': int(zero_weights),
                'total_weights': int(total_weights)
            }
            
            results['pruning_analysis'].append(analysis)
        
        return results
    
    def compare_policies(self, policy_paths: List[Path]) -> pd.DataFrame:
        """
        Compare compressibility of multiple policies.
        
        Args:
            policy_paths: List of paths to policy files
            
        Returns:
            DataFrame with comparison results
        """
        all_results = []
        
        for policy_path in policy_paths:
            # Analyze policy
            analysis = self.analyze_policy_file(policy_path)
            
            # Extract key metrics
            metrics = {
                'policy': policy_path.stem,
                'file_size_mb': analysis['file_size_bytes'] / 1e6
            }
            
            # Add compression metrics
            if 'torch_state_dict' in analysis['compression_analysis']:
                comp = analysis['compression_analysis']['torch_state_dict']
                metrics.update({
                    'compression_ratio': comp['compression_ratio'],
                    'compression_gain': comp['compression_gain']
                })
            
            # Add weight metrics
            if 'weights_only' in analysis['compression_analysis']:
                weights = analysis['compression_analysis']['weights_only']
                metrics.update({
                    'weight_entropy': weights['weight_entropy'],
                    'weight_sparsity': weights['weight_sparsity']
                })
            
            # Add parameter statistics
            if 'parameter_statistics' in analysis:
                params = analysis['parameter_statistics']
                metrics.update({
                    'total_params': params['total_parameters'],
                    'mean_abs_value': params['mean_abs_value'],
                    'param_sparsity_001': params['sparsity_001']
                })
            
            all_results.append(metrics)
        
        return pd.DataFrame(all_results)
    
    def generate_compression_report(self, policy_dir: Path, 
                                   output_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive compression analysis report.
        
        Args:
            policy_dir: Directory containing policy files
            output_dir: Directory to save report
            
        Returns:
            Dictionary with full analysis results
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all policy files
        policy_files = list(policy_dir.glob('*.pt'))
        
        if not policy_files:
            print(f"No policy files found in {policy_dir}")
            return {}
        
        print(f"Found {len(policy_files)} policy files")
        
        # Group by agent type (AEP vs reward)
        aep_policies = [p for p in policy_files if 'aep' in p.name.lower()]
        reward_policies = [p for p in policy_files if 'reward' in p.name.lower()]
        
        print(f"AEP policies: {len(aep_policies)}")
        print(f"Reward policies: {len(reward_policies)}")
        
        # Analyze all policies
        all_results = []
        pruning_results = []
        
        for policy_file in policy_files:
            print(f"\nAnalyzing {policy_file.name}...")
            
            # Basic analysis
            analysis = self.analyze_policy_file(policy_file)
            analysis['agent_type'] = 'aep' if 'aep' in policy_file.name.lower() else 'reward'
            all_results.append(analysis)
            
            # Pruning analysis (for final models only)
            if 'final' in policy_file.name.lower():
                print(f"  Running pruning analysis...")
                pruning = self.analyze_pruning_effect(policy_file)
                pruning['policy'] = policy_file.name
                pruning['agent_type'] = analysis['agent_type']
                pruning_results.append(pruning)
        
        # Compare policies
        comparison_df = self.compare_policies(policy_files)
        
        # Calculate summary statistics
        summary = {
            'total_policies_analyzed': len(policy_files),
            'aep_policies': len(aep_policies),
            'reward_policies': len(reward_policies),
            'comparison_statistics': {}
        }
        
        if len(aep_policies) > 0 and len(reward_policies) > 0:
            # Compare AEP vs reward policies
            aep_df = comparison_df[comparison_df['policy'].str.contains('aep', case=False)]
            reward_df = comparison_df[comparison_df['policy'].str.contains('reward', case=False)]
            
            if len(aep_df) > 0 and len(reward_df) > 0:
                for column in ['compression_ratio', 'compression_gain', 
                              'weight_entropy', 'weight_sparsity']:
                    if column in aep_df.columns and column in reward_df.columns:
                        aep_mean = aep_df[column].mean()
                        reward_mean = reward_df[column].mean()
                        difference = aep_mean - reward_mean
                        
                        summary['comparison_statistics'][column] = {
                            'aep_mean': float(aep_mean),
                            'reward_mean': float(reward_mean),
                            'difference': float(difference),
                            'percent_difference': float((difference / reward_mean) * 100) if reward_mean != 0 else 0
                        }
        
        # Save results
        results_file = output_dir / 'compression_analysis.json'
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': all_results,
                'comparison_dataframe': comparison_df.to_dict('records'),
                'pruning_analysis': pruning_results,
                'summary': summary
            }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        # Save comparison dataframe as CSV
        comparison_df.to_csv(output_dir / 'policy_comparison.csv', index=False)
        
        # Generate visualization
        try:
            self.generate_visualizations(comparison_df, pruning_results, output_dir)
        except Exception as e:
            print(f"Could not generate visualizations: {e}")
        
        # Print summary
        print("\n" + "="*70)
        print("COMPRESSION ANALYSIS SUMMARY")
        print("="*70)
        
        if 'comparison_statistics' in summary and summary['comparison_statistics']:
            for metric, stats in summary['comparison_statistics'].items():
                print(f"\n{metric.upper().replace('_', ' ')}:")
                print(f"  AEP mean: {stats['aep_mean']:.4f}")
                print(f"  Reward mean: {stats['reward_mean']:.4f}")
                print(f"  Difference: {stats['difference']:.4f}")
                
                # Check Prediction A1
                if metric in ['compression_gain', 'weight_sparsity']:
                    # Higher compression gain and sparsity is better for AEP
                    if stats['difference'] > 0:
                        print(f"  ✓ AEP policies have higher {metric}")
                    else:
                        print(f"  ✗ Reward policies have higher {metric}")
                elif metric == 'compression_ratio':
                    # Lower compression ratio is better (more compressible)
                    if stats['difference'] < 0:
                        print(f"  ✓ AEP policies are more compressible")
                    else:
                        print(f"  ✗ Reward policies are more compressible")
        
        print(f"\nResults saved to: {output_dir}")
        
        return summary
    
    def generate_visualizations(self, comparison_df: pd.DataFrame, 
                               pruning_results: List[Dict], 
                               output_dir: Path):
        """Generate visualization plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("colorblind")
        
        # 1. Compression comparison bar plot
        if 'compression_ratio' in comparison_df.columns and 'agent_type' in comparison_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Group by agent type
            grouped = comparison_df.groupby('agent_type')['compression_ratio'].agg(['mean', 'std', 'count'])
            
            x_pos = np.arange(len(grouped))
            plt.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
            plt.xticks(x_pos, [idx.upper() for idx in grouped.index])
            plt.ylabel('Compression Ratio (lower = better)')
            plt.title('Policy Compression Ratio by Agent Type')
            plt.grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (idx, row) in enumerate(grouped.iterrows()):
                plt.text(i, row['mean'] + 0.01, f"{row['mean']:.3f}", 
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'compression_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Weight sparsity comparison
        if 'weight_sparsity' in comparison_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Create box plot
            sns.boxplot(data=comparison_df, x='agent_type', y='weight_sparsity')
            sns.stripplot(data=comparison_df, x='agent_type', y='weight_sparsity',
                         color='black', alpha=0.5, jitter=True)
            
            plt.ylabel('Weight Sparsity (fraction near zero)')
            plt.title('Weight Sparsity Comparison')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sparsity_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Pruning analysis (if available)
        if pruning_results:
            plt.figure(figsize=(12, 8))
            
            for i, pruning in enumerate(pruning_results):
                if 'pruning_analysis' in pruning:
                    sparsity_levels = [a['target_sparsity'] for a in pruning['pruning_analysis']]
                    compression_ratios = [a['compression_ratio'] for a in pruning['pruning_analysis']]
                    
                    plt.plot(sparsity_levels, compression_ratios, 
                            'o-', label=pruning['policy'], alpha=0.7)
            
            plt.xlabel('Target Sparsity')
            plt.ylabel('Compression Ratio')
            plt.title('Pruning Effect on Compressibility')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'pruning_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Correlation matrix (if enough metrics)
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            plt.figure(figsize=(10, 8))
            
            corr_matrix = comparison_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, cbar_kws={'shrink': 0.8})
            
            plt.title('Correlation Matrix of Policy Metrics')
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze policy compressibility')
    parser.add_argument('--policies_dir', type=str, default='saved_agents',
                       help='Directory containing saved policy files')
    parser.add_argument('--output', type=str, default='compression_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare AEP vs reward policies')
    
    args = parser.parse_args()
    
    analyzer = PolicyCompressionAnalyzer()
    
    # Analyze policies
    summary = analyzer.generate_compression_report(
        Path(args.policies_dir), 
        Path(args.output)
    )
    
    # Print conclusion
    print("\n" + "="*70)
    print("PREDICTION A1: POLICY COMPRESSIBILITY")
    print("="*70)
    
    if 'comparison_statistics' in summary and summary['comparison_statistics']:
        if 'compression_ratio' in summary['comparison_statistics']:
            stats = summary['comparison_statistics']['compression_ratio']
            if stats['difference'] < 0:
                print("✓ PREDICTION A1 SUPPORTED")
                print(f"  AEP policies are more compressible than reward policies")
                print(f"  Compression ratio: AEP={stats['aep_mean']:.3f} < Reward={stats['reward_mean']:.3f}")
            else:
                print("✗ PREDICTION A1 NOT SUPPORTED")
                print(f"  AEP policies are not more compressible than reward policies")
                print(f"  Compression ratio: AEP={stats['aep_mean']:.3f} > Reward={stats['reward_mean']:.3f}")
        
        # Check other metrics
        if 'compression_gain' in summary['comparison_statistics']:
            stats = summary['comparison_statistics']['compression_gain']
            print(f"\nCompression gain: AEP={stats['aep_mean']:.3f}, Reward={stats['reward_mean']:.3f}")
        
        if 'weight_sparsity' in summary['comparison_statistics']:
            stats = summary['comparison_statistics']['weight_sparsity']
            print(f"Weight sparsity: AEP={stats['aep_mean']:.3f}, Reward={stats['reward_mean']:.3f}")
    else:
        print("Insufficient data for comparison")
        print("Need both AEP and reward policy files for comparison")

if __name__ == '__main__':
    import sys
    main()
