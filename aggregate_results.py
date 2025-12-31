#!/usr/bin/env python3
"""
Aggregate results from all experiments for meta-analysis.
Combines data from different experiments to test the overall Faith hypothesis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ResultsAggregator:
    """Aggregate and analyze results from all experiments."""
    
    def __init__(self, results_dir: str = "results_all"):
        self.results_dir = Path(results_dir)
        self.experiments = {}
        
        # Expected experiment files
        self.expected_files = {
            "speech_compression": "speech_compression/speech_compression_results.csv",
            "hrv_entropy": "hrv_entropy/hrv_results.csv",
            "social_networks": "social_networks/network_results.csv",
            "rng_entropy": "rng_entropy/rng_results.csv",
            "ai_policies": "ai_policies/ai_results.csv"
        }
    
    def load_all_results(self) -> Dict[str, pd.DataFrame]:
        """Load results from all experiments."""
        loaded = {}
        
        for exp_name, rel_path in self.expected_files.items():
            file_path = self.results_dir / rel_path
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    loaded[exp_name] = df
                    print(f"✅ Loaded {exp_name}: {len(df)} rows")
                except Exception as e:
                    print(f"❌ Failed to load {exp_name}: {e}")
            else:
                print(f"⚠️  Missing {exp_name}: {file_path}")
        
        self.experiments = loaded
        return loaded
    
    def calculate_effect_sizes(self) -> pd.DataFrame:
        """Calculate effect sizes for each experiment."""
        effects = []
        
        # Speech compression effect
        if "speech_compression" in self.experiments:
            df = self.experiments["speech_compression"]
            if "condition" in df.columns and "compression_ratio" in df.columns:
                virtue_mean = df[df["condition"] == "virtue"]["compression_ratio"].mean()
                vice_mean = df[df["condition"] == "vice"]["compression_ratio"].mean()
                pooled_std = np.sqrt((df["compression_ratio"].var() * 2) / 2)
                
                if pooled_std > 0:
                    cohens_d = (virtue_mean - vice_mean) / pooled_std
                    effects.append({
                        "experiment": "speech_compression",
                        "effect_size": cohens_d,
                        "interpretation": "Negative = virtue compresses more",
                        "hypothesis_supported": cohens_d < 0,
                        "n": len(df)
                    })
        
        # HRV entropy effect
        if "hrv_entropy" in self.experiments:
            df = self.experiments["hrv_entropy"]
            if "condition" in df.columns and "sample_entropy" in df.columns:
                # Assuming conditions: meditation (virtue) vs stress (vice)
                meditation_data = df[df["condition"].str.contains("meditation|compassion", case=False, na=False)]
                stress_data = df[df["condition"].str.contains("stress|anger", case=False, na=False)]
                
                if len(meditation_data) > 0 and len(stress_data) > 0:
                    med_mean = meditation_data["sample_entropy"].mean()
                    stress_mean = stress_data["sample_entropy"].mean()
                    
                    # Pooled standard deviation
                    n1, n2 = len(meditation_data), len(stress_data)
                    var1 = meditation_data["sample_entropy"].var(ddof=1)
                    var2 = stress_data["sample_entropy"].var(ddof=1)
                    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (med_mean - stress_mean) / pooled_std
                        effects.append({
                            "experiment": "hrv_entropy",
                            "effect_size": cohens_d,
                            "interpretation": "Negative = meditation reduces entropy",
                            "hypothesis_supported": cohens_d < 0,
                            "n": len(df)
                        })
        
        # Add similar calculations for other experiments...
        
        return pd.DataFrame(effects)
    
    def meta_analysis(self) -> Dict:
        """Perform meta-analysis across all experiments."""
        effect_df = self.calculate_effect_sizes()
        
        if len(effect_df) == 0:
            return {"error": "No effect sizes calculated"}
        
        # Calculate combined effect (simple unweighted average for now)
        # In a real meta-analysis, you'd use inverse variance weighting
        mean_effect = effect_df["effect_size"].mean()
        std_effect = effect_df["effect_size"].std()
        se_effect = std_effect / np.sqrt(len(effect_df))
        
        # Count hypotheses supported
        n_supported = effect_df["hypothesis_supported"].sum()
        n_total = len(effect_df)
        
        # Binomial test probability (if hypotheses were random)
        p_value = 1 - sum([
            np.math.comb(n_total, k) * (0.5 ** n_total)
            for k in range(n_supported)
        ])
        
        return {
            "n_experiments": n_total,
            "n_supported": int(n_supported),
            "support_rate": n_supported / n_total,
            "mean_effect_size": mean_effect,
            "se_effect_size": se_effect,
            "p_value_binomial": p_value,
            "experiment_details": effect_df.to_dict("records"),
            "interpretation": self._interpret_meta_results(n_supported, n_total, mean_effect)
        }
    
    def _interpret_meta_results(self, n_supported: int, n_total: int, mean_effect: float) -> str:
        """Interpret meta-analysis results."""
        support_rate = n_supported / n_total
        
        if support_rate >= 0.8 and mean_effect < 0:
            return "STRONG SUPPORT: Most experiments support the Faith hypothesis with negative effect sizes."
        elif support_rate >= 0.6:
            return "MODERATE SUPPORT: Majority of experiments support the Faith hypothesis."
        elif support_rate >= 0.5:
            return "WEAK SUPPORT: About half of experiments support the hypothesis."
        else:
            return "NO CLEAR SUPPORT: Less than half of experiments support the hypothesis."
    
    def create_summary_plot(self, output_path: Optional[Path] = None):
        """Create summary visualization of all results."""
        if not output_path:
            output_path = self.results_dir / "summary_plot.png"
        
        effect_df = self.calculate_effect_sizes()
        
        if len(effect_df) == 0:
            print("No effect sizes to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Forest plot of effect sizes
        colors = ['green' if supp else 'red' for supp in effect_df["hypothesis_supported"]]
        y_pos = np.arange(len(effect_df))
        
        plt.barh(y_pos, effect_df["effect_size"], color=colors, alpha=0.7)
        plt.yticks(y_pos, effect_df["experiment"])
        plt.xlabel("Effect Size (Cohen's d)")
        plt.title("Faith Hypothesis: Effect Sizes Across Experiments")
        
        # Add reference line at 0
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add text annotations
        for i, row in effect_df.iterrows():
            plt.text(row["effect_size"] + (0.01 if row["effect_size"] >= 0 else -0.05),
                     i, f"d={row['effect_size']:.2f}", 
                     va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to: {output_path}")
    
    def generate_final_report(self):
        """Generate final comprehensive report."""
        self.load_all_results()
        
        # Meta-analysis
        meta_results = self.meta_analysis()
        
        # Create report
        report = {
            "report_date": pd.Timestamp.now().isoformat(),
            "experiments_loaded": list(self.experiments.keys()),
            "meta_analysis": meta_results,
            "total_participants_approx": self._estimate_total_participants(),
            "data_quality": self._assess_data_quality(),
            "conclusions": self._generate_conclusions(meta_results)
        }
        
        # Save report
        report_path = self.results_dir / "final_meta_analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create plot
        self.create_summary_plot()
        
        # Print summary
        print("\n" + "="*70)
        print("FINAL META-ANALYSIS REPORT")
        print("="*70)
        
        if "error" not in meta_results:
            print(f"\nExperiments analyzed: {meta_results['n_experiments']}")
            print(f"Hypotheses supported: {meta_results['n_supported']}/{meta_results['n_experiments']}")
            print(f"Support rate: {meta_results['support_rate']:.1%}")
            print(f"Mean effect size: {meta_results['mean_effect_size']:.3f}")
            print(f"Binomial p-value: {meta_results['p_value_binomial']:.4f}")
            print(f"\nInterpretation: {meta_results['interpretation']}")
        
        print(f"\n📄 Full report: {report_path}")
        
        return report
    
    def _estimate_total_participants(self) -> int:
        """Estimate total number of participants across experiments."""
        total = 0
        for exp_name, df in self.experiments.items():
            if "participant_id" in df.columns:
                total += df["participant_id"].nunique()
            else:
                total += len(df)  # Rough estimate
        return total
    
    def _assess_data_quality(self) -> Dict:
        """Assess data quality across experiments."""
        quality = {}
        for exp_name, df in self.experiments.items():
            n_rows = len(df)
            n_missing = df.isnull().sum().sum()
            completeness = 1 - (n_missing / (n_rows * len(df.columns))) if n_rows > 0 else 0
            
            quality[exp_name] = {
                "n_rows": n_rows,
                "n_missing": int(n_missing),
                "completeness": completeness,
                "quality": "GOOD" if completeness > 0.9 else "FAIR" if completeness > 0.7 else "POOR"
            }
        return quality
    
    def _generate_conclusions(self, meta_results: Dict) -> List[str]:
        """Generate conclusions based on meta-analysis."""
        conclusions = []
        
        if "error" in meta_results:
            conclusions.append("Insufficient data for meta-analysis.")
            return conclusions
        
        support_rate = meta_results.get("support_rate", 0)
        mean_effect = meta_results.get("mean_effect_size", 0)
        
        if support_rate >= 0.8 and mean_effect < -0.2:
            conclusions.append("The Faith hypothesis receives strong empirical support across multiple domains.")
            conclusions.append("Virtuous/coherent states consistently show lower algorithmic entropy.")
            conclusions.append("The Anti-Entropic Principle appears to manifest in physical, biological, and computational systems.")
        elif support_rate >= 0.6:
            conclusions.append("The Faith hypothesis receives moderate empirical support.")
            conclusions.append("There is evidence for entropy reduction in virtuous/coherent states.")
            conclusions.append("Further replication with larger samples is recommended.")
        elif support_rate >= 0.5:
            conclusions.append("The Faith hypothesis receives weak or equivocal support.")
            conclusions.append("Some experiments support the hypothesis while others do not.")
            conclusions.append("More refined experimental designs may be needed.")
        else:
            conclusions.append("The Faith hypothesis is not strongly supported by the current data.")
            conclusions.append("Alternative interpretations of the results should be considered.")
        
        # Add specific recommendations
        conclusions.append("Recommendation: All experiments should be replicated independently.")
        conclusions.append("Recommendation: Pre-registration of future studies is advised.")
        conclusions.append("Recommendation: Meta-analysis should be updated as new data becomes available.")
        
        return conclusions

def main():
    parser = argparse.ArgumentParser(description="Aggregate results from all experiments")
    parser.add_argument("--results-dir", type=str, default="results_all",
                       help="Directory containing experiment results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for aggregated results")
    
    args = parser.parse_args()
    
    aggregator = ResultsAggregator(args.results_dir)
    report = aggregator.generate_final_report()
    
    print("\n" + "="*70)
    print("AGGREGATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
