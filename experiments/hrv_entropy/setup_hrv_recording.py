#!/usr/bin/env python3
"""
Setup script for HRV recording equipment.
Provides guidance for setting up different HRV monitoring systems.
"""

import sys
from pathlib import Path
import json

class HRVSetupGuide:
    """Guide for setting up HRV recording equipment."""
    
    SYSTEMS = {
        'polar_h10': {
            'name': 'Polar H10',
            'type': 'ECG chest strap',
            'sampling_rate': 130,  # Hz for RR intervals
            'software': ['Polar Beat', 'Kubios HRV', 'Elite HRV'],
            'connection': 'Bluetooth',
            'setup_steps': [
                '1. Moisten the electrodes on the chest strap',
                '2. Attach strap snugly below chest muscles',
                '3. Start Polar Beat app on smartphone',
                '4. Pair via Bluetooth',
                '5. Start recording before each condition',
                '6. Export RR intervals as CSV'
            ],
            'notes': 'Most reliable consumer-grade device. Can export raw ECG with Polar Sensor Logger app.'
        },
        
        'empatica_e4': {
            'name': 'Empatica E4',
            'type': 'Wrist-worn PPG',
            'sampling_rate': 64,  # Hz for BVP
            'software': ['E4 Connect', 'Kubios HRV'],
            'connection': 'Bluetooth/WiFi',
            'setup_steps': [
                '1. Charge device fully',
                '2. Wear on wrist (not too tight)',
                '3. Connect to E4 Connect app',
                '4. Start recording via app or API',
                '5. Export IBI (inter-beat interval) data'
            ],
            'notes': 'Convenient but less accurate than ECG. Good for longer recordings.'
        },
        
        'biopac': {
            'name': 'Biopac Systems',
            'type': 'Research-grade ECG',
            'sampling_rate': 1000,  # Hz
            'software': ['AcqKnowledge'],
            'connection': 'USB',
            'setup_steps': [
                '1. Apply electrodes in Lead II configuration',
                '2. Connect to MP160 data acquisition unit',
                '3. Configure AcqKnowledge software',
                '4. Set sampling rate to 1000 Hz',
                '5. Start recording',
                '6. Export raw ECG for R-peak detection'
            ],
            'notes': 'Gold standard for research. Requires technical expertise.'
        },
        
        'kubios': {
            'name': 'Kubios HRV',
            'type': 'Software suite',
            'sampling_rate': 'Variable',
            'software': ['Kubios HRV Premium'],
            'connection': 'Various',
            'setup_steps': [
                '1. Record ECG with any compatible device',
                '2. Import data to Kubios HRV',
                '3. Automatic R-peak detection',
                '4. Manual correction if needed',
                '5. Export RR intervals and HRV metrics',
                '6. Use artifact correction algorithms'
            ],
            'notes': 'Excellent for data processing and analysis. Commercial software.'
        }
    }
    
    def print_setup_guide(self, system: str):
        """Print setup guide for a specific system."""
        if system not in self.SYSTEMS:
            print(f"Unknown system: {system}")
            print(f"Available systems: {', '.join(self.SYSTEMS.keys())}")
            return
        
        info = self.SYSTEMS[system]
        
        print("="*60)
        print(f"SETUP GUIDE: {info['name']}")
        print("="*60)
        
        print(f"\nType: {info['type']}")
        print(f"Sampling rate: {info['sampling_rate']} Hz")
        print(f"Connection: {info['connection']}")
        print(f"Software: {', '.join(info['software'])}")
        
        print(f"\nSETUP STEPS:")
        for step in info['setup_steps']:
            print(f"  {step}")
        
        print(f"\nNOTES: {info['notes']}")
        
        print(f"\nDATA FORMAT:")
        print("  Save RR intervals as CSV with columns: 'timestamp', 'rr_interval'")
        print("  Timestamp in seconds, RR intervals in seconds")
        print("  Example filename: P001_compassion_crystal_rr.csv")
    
    def generate_config_file(self, system: str, output_path: Path):
        """Generate configuration file for a recording session."""
        if system not in self.SYSTEMS:
            print(f"Unknown system: {system}")
            return
        
        config = {
            'system': system,
            'system_info': self.SYSTEMS[system],
            'recording_settings': {
                'baseline_duration': 300,  # 5 minutes
                'intervention_duration': 600,  # 10 minutes
                'recovery_duration': 300,  # 5 minutes
                'sampling_rate': self.SYSTEMS[system]['sampling_rate'],
                'artifact_correction': 'automatic',
                'export_format': 'CSV'
            },
            'conditions': ['baseline', 'compassion', 'resentment', 'neutral', 'recovery'],
            'file_naming': {
                'pattern': '{participant_id}_{condition}_{object_type}_{date}',
                'example': 'P001_compassion_crystal_20260115'
            },
            'quality_checks': [
                'Signal strength > 90%',
                'Artifact rate < 5%',
                'Recording duration matches protocol',
                'No missing segments'
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration file saved: {output_path}")
    
    def check_prerequisites(self):
        """Check if required software is available."""
        import subprocess
        import importlib
        
        print("Checking prerequisites...")
        
        # Check Python packages
        packages = ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn']
        missing = []
        
        for package in packages:
            try:
                importlib.import_module(package)
                print(f"  ✓ {package}")
            except ImportError:
                missing.append(package)
                print(f"  ✗ {package}")
        
        if missing:
            print(f"\nMissing packages: {', '.join(missing)}")
            print("Install with: pip install " + " ".join(missing))
        
        # Check for optional HRV packages
        print("\nOptional HRV packages:")
        for package in ['heartpy', 'neurokit2', 'wfdb']:
            try:
                importlib.import_module(package)
                print(f"  ✓ {package} (optional)")
            except ImportError:
                print(f"  ✗ {package} (optional)")
        
        print("\n✓ Prerequisite check complete")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='HRV Recording Setup Guide')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Setup guide
    guide_parser = subparsers.add_parser('guide', help='Show setup guide for a system')
    guide_parser.add_argument('system', type=str, 
                            choices=['polar_h10', 'empatica_e4', 'biopac', 'kubios'],
                            help='HRV monitoring system')
    
    # Config generation
    config_parser = subparsers.add_parser('config', help='Generate configuration file')
    config_parser.add_argument('system', type=str, 
                              choices=['polar_h10', 'empatica_e4', 'biopac', 'kubios'],
                              help='HRV monitoring system')
    config_parser.add_argument('--output', type=str, default='hrv_config.json',
                              help='Output configuration file')
    
    # Check prerequisites
    subparsers.add_parser('check', help='Check prerequisites')
    
    # List systems
    subparsers.add_parser('list', help='List available systems')
    
    args = parser.parse_args()
    
    guide = HRVSetupGuide()
    
    if args.command == 'guide':
        guide.print_setup_guide(args.system)
    
    elif args.command == 'config':
        guide.generate_config_file(args.system, Path(args.output))
    
    elif args.command == 'check':
        guide.check_prerequisites()
    
    elif args.command == 'list':
        print("Available HRV monitoring systems:")
        for system_id, info in guide.SYSTEMS.items():
            print(f"\n  {system_id}:")
            print(f"    Name: {info['name']}")
            print(f"    Type: {info['type']}")
            print(f"    Sampling: {info['sampling_rate']} Hz")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
