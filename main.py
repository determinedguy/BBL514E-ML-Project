# main.py
"""
Main entry point for Network Traffic  Data Analyzer
"""

import sys
import os
import argparse

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_and_explore
from src.preprocessing import preprocess_data
from src.train import train_all_models
from src.evaluate import evaluate_all_models
from src.config import SAMPLE_SIZE

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Network Intrusion Detection System - Pattern Recognition Project'
    )
    parser.add_argument('--mode', type=str, default='explore',
                       choices=['explore', 'preprocess', 'train', 'evaluate', 'full'],
                       help='Execution mode')
    parser.add_argument('--sample', type=int, default=SAMPLE_SIZE,
                       help=f'Sample size for dataset (default: {SAMPLE_SIZE})')
    parser.add_argument('--no-sample', action='store_true',
                       help='Use full dataset (no sampling)')
    
    args = parser.parse_args()
    
    # Determine sample size
    sample_size = None if args.no_sample else args.sample
    
    print("\n" + "="*60)
    print("NETWORK TRAFFIC DATA ANALYZER")
    print("Pattern Recognition & Analysis Course Project")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Sample size: {sample_size if sample_size else 'Full dataset'}")
    print("="*60)
    
    try:
        if args.mode == 'explore':
            print("\n>>> Running data exploration...")
            df = load_and_explore(sample_size=sample_size)
            print(f"\n✓ Explored {len(df)} records")
            
        elif args.mode == 'preprocess':
            print("\n>>> Running preprocessing pipeline...")
            df = load_and_explore(sample_size=sample_size)
            preprocess_data(df)
            
        elif args.mode == 'train':
            print("\n>>> Training all models...")
            train_all_models()
            
        elif args.mode == 'evaluate':
            print("\n>>> Evaluating all models...")
            evaluate_all_models()
            
        elif args.mode == 'full':
            print("\n>>> Running FULL pipeline...")
            
            print("\n" + "="*60)
            print("[1/4] DATA EXPLORATION")
            print("="*60)
            df = load_and_explore(sample_size=sample_size)
            
            print("\n" + "="*60)
            print("[2/4] PREPROCESSING")
            print("="*60)
            preprocess_data(df)
            
            print("\n" + "="*60)
            print("[3/4] TRAINING MODELS")
            print("="*60)
            train_all_models()
            
            print("\n" + "="*60)
            print("[4/4] EVALUATING MODELS")
            print("="*60)
            evaluate_all_models()
            
            print("\n" + "="*60)
            print("✓✓✓ FULL PIPELINE COMPLETE! ✓✓✓")
            print("="*60)
            print("\n📊 Results:")
            print(f"  - Visualizations: results/")
            print(f"  - Trained models: models/")
            print(f"  - Processed data: data_processed/")
        
        print("\n" + "="*60)
        print("✓ EXECUTION COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ ERROR OCCURRED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
