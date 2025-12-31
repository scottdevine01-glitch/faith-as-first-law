#!/usr/bin/env python3
"""
Main experiment runner for AI Policies (Predictions A1-A2).
Orchestrates training, analysis, and generalization testing.
"""

import subprocess
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import shutil

def run_command(cmd: str, description: str = "") -> bool:
    """Run a shell command and print output."""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
    
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def setup_experiment(output_dir: Path):
    """Set up experiment directory structure."""
    print(f"Setting up experiment in: {output_dir}")
    
    directories = [
        output_dir,
        output_dir / 'trained_agents',
        output_dir / 'compression_analysis',
        output_dir / 'generalization_tests',
        output_dir / 'logs',
        output_dir / 'results'
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Create experiment metadata
    metadata = {
        'experiment_name': 'AI Policies A1-A2',
        'start_time': datetime.now().isoformat(),
        'predictions': ['A1', 'A2'],
        'description': 'Test whether AEP-aligned AI develops more compressible, generalizable policies'
    }
    
    with open(output_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True

def run_training_phase(output_dir: Path, config: dict) -> bool:
    """Run training phase: train AEP and reward agents."""
    print("\n" + "="*60)
    print("PHASE 1: TRAINING AGENTS")
    print("="*60)
    
    success = True
    
    # Train agents in different environments
    for env in config.get('environments', ['moral_gridworld', 'prisoner_dilemma']):
        for agent_type in ['aep', 'reward']:
            # Set hyperparameters based on agent type
            aep_lambda = 0.1 if agent_type == 'aep' else 0.0
            
            cmd = (
                f"python train_agents.py "
                f"--env {env} "
                f"--epochs {config.get('epochs', 1000)} "
                f"--hidden_dim {config.get('hidden_dim', 128)} "
                f"--output {output_dir / 'trained_agents' / f'{env}_{agent_type}'} "
            )
            
            desc = f"Training {agent_type.upper()} agent in {env}"
            if not run_command(cmd, desc):
                success = False
    
    return success

def run_compression_analysis(output_dir: Path) -> bool:
    """Run compression analysis (Prediction A1)."""
    print("\n" + "="*60)
    print("PHASE 2: POLICY COMPRESSION ANALYSIS (Prediction A1)")
    print("="*60)
    
    trained_dir = output_dir / 'trained_agents'
    analysis_dir = output_dir / 'compression_analysis'
    
    # Find all trained policy files
    policy_files = list(trained_dir.rglob('*.pt'))
    
    if not policy_files:
        print("No policy files found for analysis")
        return False
    
    print(f"Found {len(policy_files)} policy files for analysis")
    
    # Group by agent type
    aep_files = [str(f) for f in policy_files if 'aep' in f.name.lower()]
    reward_files = [str(f) for f in policy_files if 'reward' in f.name.lower()]
    
    # Run analysis on all policies
    cmd = (
        f"python policy_compression.py "
        f"--policies {trained_dir} "
        f"--pattern *.pt "
        f"--compare "
        f"--output {analysis_dir}"
    )
    
    if not run_command(cmd, "Analyzing policy compressibility"):
        return False
    
    # Also run pruning analysis
    print("\nRunning pruning sensitivity analysis...")
    for policy_file in aep_files[:2] + reward_files[:2]:  # Sample a few
        cmd = (
            f"python policy_compression.py "
            f"--policies {policy_file} "
            f"--prune 0.3 "
            f"--output {analysis_dir / 'pruning'}"
        )
        run_command(cmd, f"Pruning analysis: {Path(policy_file).name}")
    
    return True

def run_generalization_tests(output_dir: Path) -> bool:
    """Run generalization tests (Prediction A2)."""
    print("\n" + "="*60)
    print("PHASE 3: GENERALIZATION TESTING (Prediction A2)")
    print("="*60)
    
    trained_dir = output_dir / 'trained_agents'
    test_dir = output_dir / 'generalization_tests'
    
    # Find final trained agents
    aep_agents = list(trained_dir.rglob('*aep*agent_final.pt'))
    reward_agents = list(trained_dir.rglob('*reward*agent_final.pt'))
    
    if not aep_agents or not reward_agents:
        print("No trained agents found for generalization testing")
        return False
    
    # Run generalization tests
    success = True
    
    for aep_agent in aep_agents[:2]:  # Test up to 2 AEP agents
        env_name = aep_agent.parent.name.split('_')[0]
        
        # Find corresponding reward agent
        reward_agent = next((r for r in reward_agents if env_name in str(r)), None)
        
        if reward_agent:
            cmd = (
                f"python test_generalization.py "
                f"--aep_agent {aep_agent} "
                f"--reward_agent {reward_agent} "
                f"--test_scenarios novel "
                f"--n_tests 100 "
                f"--output {test_dir / env_name}"
            )
            
            desc = f"Generalization test: {env_name}"
            if not run_command(cmd, desc):
                success = False
    
    return success

def generate_final_report(output_dir: Path):
    """Generate final experiment report."""
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)
    
    report = {
        'experiment': 'AI Policies A1-A2',
        'completion_time': datetime.now().isoformat(),
        'predictions_tested': ['A1', 'A2'],
        'results': {}
    }
    
    # Load compression analysis results
    compression_file = output_dir / 'compression_analysis' / 'policy_comparison.json'
    if compression_file.exists():
        with open(compression_file, 'r') as f:
            compression_results = json.load(f)
        
        report['results']['prediction_a1'] = {
            'description': 'AEP policies are more compressible than reward policies',
            'compression_analysis': compression_results.get('group_comparison', {})
        }
        
        # Check Prediction A1
        if 'compression_ratio' in compression_results.get('group_comparison', {}):
            ratio = compression_results['group_comparison']['compression_ratio']
            report['results']['prediction_a1']['supported'] = ratio < 1
            report['results']['prediction_a1']['compression_ratio'] = ratio
    
    # Load generalization test results
    generalization_files = list((output_dir / 'generalization_tests').rglob('*results.json'))
    if generalization_files:
        generalization_results = []
        for file in generalization_files:
            with open(file, 'r') as f:
                results = json.load(f)
                generalization_results.append(results)
        
        report['results']['prediction_a2'] = {
            'description': 'AEP agents generalize better to novel scenarios',
            'generalization_tests': generalization_results
        }
        
        # Check Prediction A2
        aep_better_count = sum(1 for r in generalization_results 
                              if r.get('aep_better', False))
        report['results']['prediction_a2']['supported'] = aep_better_count > len(generalization_results) / 2
        report['results']['prediction_a2']['aep_better_percentage'] = (
            aep_better_count / len(generalization_results) * 100
            if generalization_results else 0
        )
    
    # Save final report
    report_file = output_dir / 'final_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable summary
    summary_file = output_dir / 'experiment_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("AI POLICIES EXPERIMENT SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("PREDICTION A1: Policy Compressibility\n")
        f.write("-"*40 + "\n")
        if 'prediction_a1' in report['results']:
            a1 = report['results']['prediction_a1']
            f.write(f"Description: {a1['description']}\n")
            if 'supported' in a1:
                f.write(f"Supported: {'YES' if a1['supported'] else 'NO'}\n")
            if 'compression_ratio' in a1:
                f.write(f"Compression Ratio (AEP/Reward): {a1['compression_ratio']:.3f}\n")
                if a1['compression_ratio'] < 1:
                    f.write("✓ AEP policies are more compressible\n")
                else:
                    f.write("✗ AEP policies are NOT more compressible\n")
        
        f.write("\nPREDICTION A2: Generalization\n")
        f.write("-"*40 + "\n")
        if 'prediction_a2' in report['results']:
            a2 = report['results']['prediction_a2']
            f.write(f"Description: {a2['description']}\n")
            if 'supported' in a2:
                f.write(f"Supported: {'YES' if a2['supported'] else 'NO'}\n")
            if 'aep_better_percentage' in a2:
                f.write(f"AEP better in: {a2['aep_better_percentage']:.1f}% of tests\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("EXPERIMENT COMPLETE\n")
        f.write(f"Results saved to: {output_dir}\n")
    
    # Print summary
    print("\nEXPERIMENT SUMMARY:")
    print("-"*40)
    if 'prediction_a1' in report['results']:
        a1 = report['results']['prediction_a1']
        print(f"Prediction A1 (Compressibility): {'SUPPORTED' if a1.get('supported', False) else 'NOT SUPPORTED'}")
    
    if 'prediction_a2' in report['results']:
        a2 = report['results']['prediction_a2']
        print(f"Prediction A2 (Generalization): {'SUPPORTED' if a2.get('supported', False) else 'NOT SUPPORTED'}")
    
    print(f"\nDetailed report: {report_file}")
    print(f"Human-readable summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Run complete AI Policies experiment (Predictions A1-A2)'
    )
    
    parser.add_argument('--output', type=str, default='ai_policies_experiment',
                       help='Output directory for experiment results')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Training epochs per agent')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training phase (use pre-trained agents)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip compression analysis')
    parser.add_argument('--skip-generalization', action='store_true',
                       help='Skip generalization tests')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'epochs': args.epochs,
        'hidden_dim': 128,
        'environments': ['moral_gridworld', 'prisoner_dilemma']
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # Setup experiment directory
    output_dir = Path(args.output)
    if not setup_experiment(output_dir):
        print("Failed to set up experiment directory")
        return 1
    
    # Save configuration
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    success = True
    
    # Run phases
    if not args.skip_training:
        if not run_training_phase(output_dir, config):
            print("Warning: Training phase had issues")
            success = False
    
    if not args.skip_analysis:
        if not run_compression_analysis(output_dir):
            print("Warning: Compression analysis had issues")
            success = False
    
    if not args.skip_generalization:
        if not run_generalization_tests(output_dir):
            print("Warning: Generalization tests had issues")
            success = False
    
    # Generate final report
    generate_final_report(output_dir)
    
    if success:
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED WITH WARNINGS")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
