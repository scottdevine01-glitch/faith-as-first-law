#!/usr/bin/env python3
"""
Test script for the speech compression analysis.
Generates synthetic data to test the analysis pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analysis module
sys.path.append(str(Path(__file__).parent))

def generate_test_data(n_participants=20) -> pd.DataFrame:
    """
    Generate synthetic test data that follows the hypothesis pattern.
    """
    np.random.seed(42)  # For reproducibility
    
    data = []
    
    for pid in range(1, n_participants + 1):
        participant_id = f"P{pid:03d}"
        
        # Generate data that follows the hypothesis: virtue < neutral < vice
        # Base compression ratios with some noise
        base_virtue = 0.40 + np.random.normal(0, 0.05)
        base_neutral = 0.50 + np.random.normal(0, 0.05)
        base_vice = 0.60 + np.random.normal(0, 0.05)
        
        # Add participant-specific baseline
        participant_baseline = np.random.normal(0, 0.02)
        
        # Conditions with the predicted pattern
        conditions = [
            ('virtue', base_virtue + participant_baseline),
            ('neutral', base_neutral + participant_baseline),
            ('vice', base_vice + participant_baseline)
        ]
        
        # Randomize order for each participant
        np.random.shuffle(conditions)
        
        for condition, cr in conditions:
            data.append({
                'participant_id': participant_id,
                'condition': condition,
                'compression_ratio': max(0.1, min(0.9, cr)),  # Keep in reasonable range
                'original_size_bytes': np.random.randint(5000, 15000),
                'compressed_size_bytes': 0,
                'word_count': np.random.randint(150, 300)
            })
    
    df = pd.DataFrame(data)
    
    # Calculate compressed size from compression ratio
    df['compressed_size_bytes'] = (df['compression_ratio'] * df['original_size_bytes']).astype(int)
    
    return df

def run_tests():
    """Run a complete test of the analysis pipeline."""
    print("Testing speech compression analysis pipeline...")
    print("="*60)
    
    # Generate test data
    test_df = generate_test_data(n_participants=10)
    
    # Save test data
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / 'test_results.csv'
    test_df.to_csv(test_file, index=False)
    print(f"✓ Generated test data: {test_file}")
    print(f"  Participants: {test_df['participant_id'].nunique()}")
    print(f"  Observations: {len(test_df)}")
    
    # Import and run analysis
    try:
        import analysis
        
        # Test descriptive statistics
        print("\n✓ Testing descriptive statistics...")
        desc_stats = analysis.descriptive_statistics(test_df)
        print(desc_stats)
        
        # Test hypothesis check
        print("\n✓ Testing hypothesis check...")
        hypothesis = analysis.check_hypothesis_pattern(test_df)
        print(f"Pattern supported: {hypothesis['pattern_full']}")
        
        # Test ANOVA
        print("\n✓ Testing ANOVA...")
        anova_results = analysis.repeated_measures_anova(test_df)
        if 'error' not in anova_results:
            print(f"F-statistic: {anova_results['f_statistic']:.3f}")
            print(f"p-value: {anova_results['p_value']:.6f}")
        
        # Test pairwise comparisons
        print("\n✓ Testing pairwise comparisons...")
        pairwise = analysis.pairwise_comparisons(test_df)
        print(pairwise[['comparison', 'p_value_adj', 'significant']].to_string())
        
        # Test visualization
        print("\n✓ Testing visualization...")
        analysis.create_visualizations(test_df, test_dir)
        print(f"Visualizations saved to: {test_dir}")
        
        # Run full report
        print("\n" + "="*60)
        print("RUNNING FULL ANALYSIS REPORT")
        print("="*60)
        
        report = analysis.generate_report(test_df, test_dir / 'full_report')
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("✓ All tests passed!")
        print(f"\nTest files saved in: {test_dir}")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
