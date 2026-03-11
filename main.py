# main.py
"""
Main entry point for Network Traffic Data Analyzer
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
from src.predict import predict_from_csv, predict_from_pcap
from src.config import SAMPLE_SIZE

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Network Traffic Data Analyzer - Pattern Recognition Project'
    )
    parser.add_argument('--mode', type=str, default='explore',
                       choices=['explore', 'preprocess', 'train', 'evaluate', 'predict', 'full'],
                       help='Execution mode')
    parser.add_argument('--sample', type=int, default=SAMPLE_SIZE,
                       help=f'Sample size for dataset (default: {SAMPLE_SIZE})')
    parser.add_argument('--no-sample', action='store_true',
                       help='Use full dataset (no sampling)')
    
    # Prediction arguments
    parser.add_argument('--input', type=str,
                       help='Input file for prediction (CSV or PCAP)')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['rf', 'mlp', 'svm', 'knn', 'ensemble'],
                       help='Model to use for prediction (default: ensemble)')
    parser.add_argument('--output', type=str,
                       help='Output file for predictions (CSV)')
    
    args = parser.parse_args()
    
    # Determine sample size
    sample_size = None if args.no_sample else args.sample
    
    print("\n" + "="*60)
    print("NETWORK TRAFFIC DATA ANALYZER")
    print("Pattern Recognition & Analysis Course Project")
    print("="*60)
    print(f"Mode: {args.mode}")
    if args.mode != 'predict':
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
        
        elif args.mode == 'predict':
            if not args.input:
                print("\n✗ Error: --input file is required for prediction mode")
                print("\nUsage:")
                print("  python main.py --mode predict --input yourfile.csv --model ensemble")
                sys.exit(1)
            
            print(f"\n>>> Running prediction on: {args.input}")
            print(f">>> Using model: {args.model}")
            
            # Detect file type
            if args.input.endswith('.pcap'):
                results = predict_from_pcap(args.input, args.model)
            elif args.input.endswith('.csv'):
                results = predict_from_csv(args.input, args.model)
            else:
                print("\n✗ Error: Input file must be .csv or .pcap")
                sys.exit(1)
            
            # Save results if output specified
            if args.output:
                results.to_csv(args.output, index=False)
                print(f"\n✓ Results saved to: {args.output}")
            else:
                # Display first 10 predictions
                print("\n" + "="*60)
                print("SAMPLE PREDICTIONS (first 10)")
                print("="*60)
                print(results.head(10))
                print("\nTo save results, use: --output results.csv")
            
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
