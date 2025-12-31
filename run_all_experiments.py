#!/usr/bin/env python3
"""
Run all experiments from the Faith as the First Law research program.
This script orchestrates the complete experimental pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

class ExperimentRunner:
    """Orchestrate running of all experiments."""
    
    def __init__(self, output_dir="results_all"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Experiment configurations
        self.experiments = {
            "speech_compression": {
                "script": "experiments/speech_compression/run_speech_study.py",
                "description": "Prediction B1: Virtuous speech compression",
                "example_cmd": "--example",
                "output": "results/speech_compression_results.csv"
            },
            "hrv_entropy": {
                "script": "experiments/hrv_entropy/run_hrv_study.py",
                "description": "Predictions B2, M1: HRV coherence",
                "example_cmd": "--test",
                "output": "results/hrv_results.csv"
            },
            "social_networks": {
                "script": "experiments/social_networks/analyze_networks.py",
                "description": "Prediction B3: Network compressibility",
                "example_cmd": "--example",
                "output": "results/network_results.csv"
            },
            "rng_entropy": {
                "script": "experiments/rng_entropy/run_rng_experiment.py",
                "description": "Prediction M3: RNG entropy reduction",
                "example_cmd": "--simulate",
                "output": "results/rng_results.csv"
            },
            "ai_policies": {
                "script": "experiments/ai_policies/train_ai_agents.py",
                "description": "Predictions A1-A2: AI policy compression",
                "example_cmd": "--quick-test",
                "output": "results/ai_results.csv"
            }
        }
    
    def run_experiment(self, name: str, use_example: bool = True) -> dict:
        """Run a single experiment."""
        config = self.experiments.get(name)
        if not config:
            return {"error": f"Unknown experiment: {name}"}
        
        print(f"\n{'='*60}")
        print(f"RUNNING: {config['description']}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [sys.executable, config["script"]]
        if use_example and config.get("example_cmd"):
            cmd.append(config["example_cmd"])
        
        # Add output directory
        cmd.extend(["--output", str(self.output_dir / name)])
        
        # Run experiment
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check for output file
            output_path = self.output_dir / name / Path(config["output"]).name
            has_output = output_path.exists()
            
            return {
                "experiment": name,
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "has_output": has_output,
                "output_path": str(output_path) if has_output else None
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "experiment": name,
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode
            }
    
    def run_all(self, use_example: bool = True) -> pd.DataFrame:
        """Run all experiments."""
        results = []
        
        for name in self.experiments:
            print(f"\n📊 Starting experiment: {name}")
            result = self.run_experiment(name, use_example)
            results.append(result)
            
            if result["success"]:
                print(f"  ✅ Success: {name}")
                if result.get("has_output"):
                    print(f"  📁 Output: {result['output_path']}")
            else:
                print(f"  ❌ Failed: {name}")
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        # Create summary
        df = pd.DataFrame(results)
        summary_file = self.output_dir / "experiment_summary.csv"
        df.to_csv(summary_file, index=False)
        
        # Generate report
        self.generate_report(df)
        
        return df
    
    def generate_report(self, results_df: pd.DataFrame):
        """Generate a summary report of all experiments."""
        successful = results_df[results_df["success"]]
        failed = results_df[~results_df["success"]]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(results_df),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results_df) if len(results_df) > 0 else 0,
            "experiments": results_df[["experiment", "success"]].to_dict("records"),
            "failed_details": failed[["experiment", "error"]].to_dict("records") if len(failed) > 0 else []
        }
        
        # Save report
        report_file = self.output_dir / "experiment_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {len(results_df)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success rate: {report['success_rate']:.1%}")
        
        if len(failed) > 0:
            print(f"\nFailed experiments:")
            for _, row in failed.iterrows():
                print(f"  ❌ {row['experiment']}: {row.get('error', 'Unknown error')}")
        
        print(f"\n📄 Full report: {report_file}")
        print(f"📊 Summary CSV: {self.output_dir / 'experiment_summary.csv'}")

def main():
    parser = argparse.ArgumentParser(
        description="Run all Faith as the First Law experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with example data
  python run_all_experiments.py --all
  
  # Run specific experiments
  python run_all_experiments.py --experiment speech_compression --experiment hrv_entropy
  
  # Run with real data (not examples)
  python run_all_experiments.py --all --no-example
        """
    )
    
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--experiment", action="append",
                       choices=["speech_compression", "hrv_entropy", 
                                "social_networks", "rng_entropy", "ai_policies"],
                       help="Run specific experiment(s)")
    parser.add_argument("--no-example", action="store_true",
                       help="Use real data instead of examples (requires data files)")
    parser.add_argument("--output", type=str, default="results_all",
                       help="Output directory for all results")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.output)
    
    if args.all:
        # Run all experiments
        print("Running all experiments...")
        runner.run_all(use_example=not args.no_example)
    
    elif args.experiment:
        # Run specific experiments
        results = []
        for exp_name in args.experiment:
            result = runner.run_experiment(exp_name, use_example=not args.no_example)
            results.append(result)
        
        # Create summary
        df = pd.DataFrame(results)
        summary_file = Path(args.output) / "selected_experiments_summary.csv"
        df.to_csv(summary_file, index=False)
        
        print(f"\nSelected experiments completed.")
        print(f"Summary saved to: {summary_file}")
    
    else:
        parser.print_help()
        print("\nNo experiments specified. Use --all or --experiment.")

if __name__ == "__main__":
    main()
