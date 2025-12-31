#!/usr/bin/env python3
"""
Policy compression analysis for Prediction A1.
Tests whether AEP-aligned AI policies are more compressible than reward-maximizing policies.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import gzip
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class PolicyCompressionAnalyzer:
    """Analyze compressibility of neural network policies."""
    
    def __init__(self):
        self.compression_methods = ['gzip', 'bz2', 'lzma']
        
    def analyze_policy_file(self, policy_path: Path) -> Dict[str, Any]:
        """
        Analyze compressibility of a saved policy file.
        
        Args:
            policy_path: Path to saved policy (.pt file)
            
        Returns:
            Dictionary with compression analysis results
        """
        # Load policy
        checkpoint = torch.load(policy_path, map_location='cpu')
        
        # Get policy state dict
        if 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
        else:
            state_dict = checkpoint
        
        # Analyze raw file compressibility
        raw_analysis = self._analyze_file_compressibility(policy_path)
        
        # Analyze model architecture compressibility
        model_analysis = self._analyze_model_compressibility(state_dict)
        
        # Analyze weight distribution
        weight_analysis = self._analyze_weight_distribution(state_dict)
        
        # Combine results
        results = {
            'policy_file': str(policy_path),
            'file_size_bytes': policy_path.stat().st_size,
            'raw_compressibility': raw_analysis,
            'model_compressibility': model_analysis,
            'weight_analysis': weight_analysis,
            'total_parameters': sum(p.numel() for p in state_dict.values()),
            'parameter_layers': len(state_dict)
        }
        
        return results
    
    def _analyze_file_compressibility(self, file_path: Path) -> Dict[str, float]:
        """Analyze compressibility of the raw file."""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        results = {'original_size': len(data)}
        
        # Gzip compression
        compressed_gzip = gzip.compress(data, compresslevel=9)
        results['gzip_size'] = len(compressed_gzip)
        results['gzip_ratio'] = len(compressed_gzip) / len(data)
        
        # Try other compressions
        try:
            import bz2
            compressed_bz2 = bz2.compress(data, compresslevel=9)
            results['bz2_size'] = len(compressed_bz2)
            results['bz2_ratio'] = len(compressed_bz2) / len(data)
        except:
            pass
        
        try:
            import lzma
            compressed_lzma = lzma.compress(data, preset=9)
            results['lzma_size'] = len(compressed_lzma)
            results['lzma_ratio'] = len(compressed_lzma) / len(data)
        except:
            pass
        
        return results
    
    def _analyze_model_compressibility(self, state_dict: Dict) -> Dict[str, Any]:
        """Analyze compressibility of model weights."""
        results = {}
        
        # Serialize weights in different formats
        weights_list = []
        for name, param in state_dict.items():
            if 'weight' in name:
                weights_list.append(param.numpy().flatten())
        
        all_weights = np.concatenate(weights_list) if weights_list else np.array([])
        
        # Analyze different serialization methods
        serializations = {}
        
        # 1. Raw binary
        raw_binary = all_weights.astype(np.float32).tobytes()
        serializations['raw_binary'] = {
            'size': len(raw_binary),
            'gzip_ratio': len(gzip.compress(raw_binary, compresslevel=9)) / len(raw_binary)
        }
        
        # 2. Pickle
        pickle_data = pickle.dumps(all_weights, protocol=pickle.HIGHEST_PROTOCOL)
        serializations['pickle'] = {
            'size': len(pickle_data),
            'gzip_ratio': len(gzip.compress(pickle_data, compresslevel=9)) / len(pickle_data)
        }
        
        # 3. JSON (with quantization)
        quantized = np.round(all_weights * 1000).astype(int)  # Quantize to 3 decimal places
        json_data = json.dumps(quantized.tolist()).encode('utf-8')
        serializations['json_quantized'] = {
            'size': len(json_data),
            'gzip_ratio': len(gzip.compress(json_data, compresslevel=9)) / len(json_data)
        }
        
        results['serialization_analysis'] = serializations
        
        # Calculate best compression ratio
        compression_ratios = [s['gzip_ratio'] for s in serializations.values()]
        results['best_compression_ratio'] = min(compression_ratios) if compression_ratios else 1.0
        results['worst_compression_ratio'] = max(compression_ratios) if compression_ratios else 1.0
        results['mean_compression_ratio'] = np.mean(compression_ratios) if compression_ratios else 1.0
        
        return results
    
    def _analyze_weight_distribution(self, state_dict: Dict) -> Dict[str, Any]:
        """Analyze weight distribution and sparsity."""
        weights = []
        for name, param in state_dict.items():
            if 'weight' in name and param.dim() >= 2:  # Only weight matrices
                weights.append(param.numpy().flatten())
        
        if not weights:
            return {'error': 'No weight parameters found'}
        
        all_weights = np.concatenate(weights)
        
        # Basic statistics
        results = {
            'n_weights': len(all_weights),
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights)),
            'abs_mean': float(np.mean(np.abs(all_weights))),
            'abs_std': float(np.std(np.abs(all_weights)))
        }
        
        # Sparsity analysis
        threshold = 0.01 * np.abs(all_weights).max()  # 1% of max absolute value
        near_zero = np.abs(all_weights) < threshold
        results['sparsity_fraction'] = float(np.mean(near_zero))
        results['significant_weights'] = int(np.sum(~near_zero))
        
        # Entropy of weight distribution
        hist, bins = np.histogram(all_weights, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        results['weight_entropy'] = float(entropy)
        
        # Kurtosis and skewness
        from scipy import stats
        results['kurtosis'] = float(stats.kurtosis(all_weights))
        results['skewness'] = float(stats.skew(all_weights))
        
        # Norm ratios
        l1_norm = np.sum(np.abs(all_weights))
        l2_norm = np.sqrt(np.sum(all_weights ** 2))
        results['l1_norm'] = float(l1_norm)
        results['l2_norm'] = float(l2_norm)
        results['l1_l2_ratio'] = float(l1_norm / l2_norm) if l2_norm > 0 else 0
        
        return results
    
    def compare_policies(self, policy_paths: List[Path]) -> Dict[str, Any]:
        """
        Compare compressibility of multiple policies.
        
        Args:
            policy_paths: List of paths to policy files
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for policy_path in policy_paths:
            if not policy_path.exists():
                print(f"Warning: {policy_path} not found, skipping")
                continue
            
            print(f"Analyzing {policy_path.name}...")
            analysis = self.analyze_policy_file(policy_path)
            results[policy_path.stem] = analysis
        
        # Generate comparison metrics
        comparison = self._generate_comparison_metrics(results)
        
        return {
            'individual_analyses': results,
            'comparison': comparison
        }
    
    def _generate_comparison_metrics(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comparison metrics between policies."""
        if len(results) < 2:
            return {'error': 'Need at least 2 policies for comparison'}
        
        comparison = {
            'policy_names': list(results.keys()),
            'compression_ratios': {},
            'parameter_counts': {},
            'weight_entropies': {},
            'sparsity_levels': {}
        }
        
        for name, analysis in results.items():
            # Best compression ratio
            comp_ratio = analysis['model_compressibility']['best_compression_ratio']
            comparison['compression_ratios'][name] = comp_ratio
            
            # Parameter count
            comparison['parameter_counts'][name] = analysis['total_parameters']
            
            # Weight entropy
            if 'weight_entropy' in analysis['weight_analysis']:
                comparison['weight_entropies'][name] = analysis['weight_analysis']['weight_entropy']
            
            # Sparsity
            if 'sparsity_fraction' in analysis['weight_analysis']:
                comparison['sparsity_levels'][name] = analysis['weight_analysis']['sparsity_fraction']
        
        # Calculate rankings
        comparison['compression_ranking'] = self._rank_by_value(
            comparison['compression_ratios'], ascending=True
        )
        comparison['entropy_ranking'] = self._rank_by_value(
            comparison['weight_entropies'], ascending=True
        ) if comparison['weight_entropies'] else {}
        
        return comparison
    
    def _rank_by_value(self, values: Dict[str, float], ascending: bool = True) -> Dict[str, int]:
        """Rank policies by a metric."""
        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=not ascending)
        return {name: rank + 1 for rank, (name, _) in enumerate(sorted_items)}
    
    def prune_and_analyze(self, 
                         policy_path: Path,
                         pruning_method: str = 'l1_unstructured',
                         amount: float = 0.3) -> Dict[str, Any]:
        """
        Prune a policy and analyze compressibility changes.
        
        Args:
            policy_path: Path to policy file
            pruning_method: Pruning method ('l1_unstructured', 'random_unstructured', etc.)
            amount: Fraction of weights to prune
            
        Returns:
            Dictionary with pruning analysis results
        """
        # Load model
        checkpoint = torch.load(policy_path, map_location='cpu')
        
        if 'policy_state_dict' not in checkpoint:
            return {'error': 'Checkpoint does not contain policy_state_dict'}
        
        # Create a simple model to hold weights
        class SimpleModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                for name, tensor in state_dict.items():
                    if 'weight' in name:
                        # Create parameter with same shape
                        param = nn.Parameter(tensor.clone())
                        self.register_parameter(name, param)
        
        model = SimpleModel(checkpoint['policy_state_dict'])
        
        # Analyze before pruning
        before_analysis = self.analyze_policy_file(policy_path)
        
        # Apply pruning
        for name, module in model.named_parameters():
            if 'weight' in name:
                if pruning_method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=amount)
                elif pruning_method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=amount)
                elif pruning_method == 'ln_structured':
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        
        # Make pruning permanent
        for name, module in model.named_parameters():
            if 'weight' in name:
                prune.remove(module, 'weight')
        
        # Save pruned model
        pruned_path = policy_path.parent / f"{policy_path.stem}_pruned_{int(amount*100)}.pt"
        torch.save({
            'policy_state_dict': dict(model.named_parameters()),
            'pruning_method': pruning_method,
            'pruning_amount': amount
        }, pruned_path)
        
        # Analyze after pruning
        after_analysis = self.analyze_policy_file(pruned_path)
        
        # Calculate improvements
        compression_improvement = (
            after_analysis['model_compressibility']['best_compression_ratio'] -
            before_analysis['model_compressibility']['best_compression_ratio']
        )
        
        sparsity_improvement = (
            after_analysis['weight_analysis']['sparsity_fraction'] -
            before_analysis['weight_analysis']['sparsity_fraction']
        )
        
        return {
            'original_policy': before_analysis,
            'pruned_policy': after_analysis,
            'pruning_details': {
                'method': pruning_method,
                'amount': amount,
                'pruned_file': str(pruned_path)
            },
            'improvements': {
                'compression_ratio_change': float(compression_improvement),
                'sparsity_change': float(sparsity_improvement),
                'parameter_reduction': float(
                    (before_analysis['total_parameters'] - after_analysis['total_parameters']) /
                    before_analysis['total_parameters']
                )
            }
        }

def analyze_policy_directory(directory: Path, 
                            pattern: str = "*.pt") -> Dict[str, Any]:
    """
    Analyze all policies in a directory.
    
    Args:
        directory: Directory containing policy files
        pattern: File pattern to match
        
    Returns:
        Dictionary with analysis of all policies
    """
    analyzer = PolicyCompressionAnalyzer()
    policy_files = list(directory.glob(pattern))
    
    if not policy_files:
        return {'error': f'No policy files found matching {pattern} in {directory}'}
    
    print(f"Found {len(policy_files)} policy files")
    
    # Group by agent type
    aep_policies = [f for f in policy_files if 'aep' in f.name.lower()]
    reward_policies = [f for f in policy_files if 'reward' in f.name.lower()]
    
    results = {
        'directory': str(directory),
        'total_policies': len(policy_files),
        'aep_policies': len(aep_policies),
        'reward_policies': len(reward_policies)
    }
    
    # Analyze all policies
    all_analysis = analyzer.compare_policies(policy_files)
    results['all_analyses'] = all_analysis
    
    # Compare AEP vs Reward if we have both
    if aep_policies and reward_policies:
        aep_analysis = analyzer.compare_policies(aep_policies)
        reward_analysis = analyzer.compare_policies(reward_policies)
        
        # Calculate group statistics
        aep_compression_ratios = [
            a['model_compressibility']['best_compression_ratio']
            for a in aep_analysis['individual_analyses'].values()
        ]
        
        reward_compression_ratios = [
            a['model_compressibility']['best_compression_ratio']
            for a in reward_analysis['individual_analyses'].values()
        ]
        
        results['group_comparison'] = {
            'aep_mean_compression': float(np.mean(aep_compression_ratios)),
            'aep_std_compression': float(np.std(aep_compression_ratios)),
            'reward_mean_compression': float(np.mean(reward_compression_ratios)),
            'reward_std_compression': float(np.std(reward_compression_ratios)),
            'compression_difference': float(
                np.mean(aep_compression_ratios) - np.mean(reward_compression_ratios)
            ),
            'compression_ratio': float(
                np.mean(aep_compression_ratios) / np.mean(reward_compression_ratios)
            ),
            'aep_entropies': [
                a['weight_analysis'].get('weight_entropy', 0)
                for a in aep_analysis['individual_analyses'].values()
            ],
            'reward_entropies': [
                a['weight_analysis'].get('weight_entropy', 0)
                for a in reward_analysis['individual_analyses'].values()
            ]
        }
        
        # Statistical test
        from scipy import stats
        if len(aep_compression_ratios) > 1 and len(reward_compression_ratios) > 1:
            t_stat, p_value = stats.ttest_ind(
                aep_compression_ratios, reward_compression_ratios, equal_var=False
            )
            results['group_comparison']['t_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze policy compressibility')
    parser.add_argument('--policies', type=str, required=True,
                       help='Directory containing policy files or specific policy file')
    parser.add_argument('--pattern', type=str, default='*.pt',
                       help='File pattern to match (default: *.pt)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare AEP vs reward policies')
    parser.add_argument('--prune', type=float, default=None,
                       help='Test pruning with specified amount (0.0-1.0)')
    parser.add_argument('--output', type=str, default='compression_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    policies_path = Path(args.policies)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    analyzer = PolicyCompressionAnalyzer()
    
    if policies_path.is_file():
        # Single file analysis
        print(f"Analyzing single policy: {policies_path}")
        results = analyzer.analyze_policy_file(policies_path)
        
        # Save results
        output_file = output_dir / f"{policies_path.stem}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        print(f"Results saved to: {output_file}")
        
        # Test pruning if requested
        if args.prune is not None:
            print(f"\nTesting pruning with amount={args.prune}")
            pruning_results = analyzer.prune_and_analyze(policies_path, amount=args.prune)
            
            pruning_file = output_dir / f"{policies_path.stem}_pruning_{int(args.prune*100)}.json"
            with open(pruning_file, 'w') as f:
                json.dump(pruning_results, f, indent=2, default=float)
            
            print(f"Pruning analysis saved to: {pruning_file}")
            
            # Print pruning improvements
            impr = pruning_results['improvements']
            print(f"Compression ratio change: {impr['compression_ratio_change']:.4f}")
            print(f"Sparsity increase: {impr['sparsity_change']:.4f}")
            print(f"Parameter reduction: {impr['parameter_reduction']:.2%}")
    
    else:
        # Directory analysis
        print(f"Analyzing policies in directory: {policies_path}")
        results = analyze_policy_directory(policies_path, args.pattern)
        
        # Save results
        output_file = output_dir / 'policy_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        print(f"Results saved to: {output_file}")
        
        # Print summary
        if 'group_comparison' in results:
            gc = results['group_comparison']
            print("\n" + "="*60)
            print("AEP vs REWARD POLICY COMPARISON")
            print("="*60)
            print(f"AEP policies: {results['aep_policies']}")
            print(f"Reward policies: {results['reward_policies']}")
            print(f"\nCompression Ratios (lower is better):")
            print(f"  AEP mean: {gc['aep_mean_compression']:.4f} ± {gc['aep_std_compression']:.4f}")
            print(f"  Reward mean: {gc['reward_mean_compression']:.4f} ± {gc['reward_std_compression']:.4f}")
            print(f"  Difference: {gc['compression_difference']:.4f}")
            print(f"  Ratio (AEP/Reward): {gc['compression_ratio']:.3f}")
            
            if gc.get('compression_ratio', 1) < 1:
                print("✓ AEP policies are more compressible (Prediction A1 supported)")
            else:
                print("✗ AEP policies are not more compressible (Prediction A1 not supported)")
            
            if 't_test' in gc:
                tt = gc['t_test']
                print(f"\nStatistical test:")
                print(f"  t({results['aep_policies']+results['reward_policies']-2}) = {tt['t_statistic']:.3f}")
                print(f"  p = {tt['p_value']:.4f}")
                if tt['significant']:
                    print("  ✓ Significant difference (p < 0.05)")
                else:
                    print("  ✗ Not statistically significant")

if __name__ == '__main__':
    main()
