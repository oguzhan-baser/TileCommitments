"""
Standalone script to compute ESD metrics for new models.

This script can be used to pre-compute metrics for any HuggingFace model
that supports the AutoModelForCausalLM interface and follows LLaMA/OPT architecture.
"""

import argparse
import os
import numpy as np
import time
from lib.esd_utils import get_esd_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Compute ESD metrics for a new model and save to .npy file'
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model identifier (HuggingFace Hub name or local path)')
    parser.add_argument('--metric', type=str, default='alpha_peak',
                       choices=['alpha_peak', 'alpha_mid', 'entropy', 'alpha', 'mp_softrank', 
                               'stable_rank', 'random_distance', 'log_norm', 'log_spectral_norm',
                               'alpha_weighted', 'log_alpha_norm', 'spectral_norm'],
                       help='ESD metric to compute')
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                       help='Directory to cache model weights')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the computed metrics (.npy file)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing metric file if it exists')
    
    args = parser.parse_args()
    
    # Check if output already exists
    output_path = os.path.join(args.output_dir, f"{args.metric}.npy")
    if os.path.exists(output_path) and not args.overwrite:
        print(f"✗ Metric file already exists: {output_path}")
        print(f"  Use --overwrite to recompute")
        print(f"\nLoading existing file to show info...")
        existing_metrics = np.load(output_path)
        print(f"  Shape: {existing_metrics.shape}")
        print(f"  Min: {existing_metrics.min():.4f}, Max: {existing_metrics.max():.4f}")
        print(f"  Mean: {existing_metrics.mean():.4f}")
        return
    
    print("="*70)
    print("COMPUTING ESD METRICS FOR NEW MODEL")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Metric: {args.metric}")
    print(f"Output: {output_path}")
    print("\nNote: This may take 5-60 minutes depending on model size...")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Compute metrics
        print("\nLoading model and computing metrics...")
        metrics = get_esd_metrics(args.model, args.metric, args.cache_dir)
        
        elapsed = time.time() - start_time
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(output_path, metrics)
        
        # Print summary
        print("\n" + "="*70)
        print("COMPUTATION COMPLETE")
        print("="*70)
        print(f"✓ Computed {len(metrics)} metric values")
        print(f"✓ Saved to {output_path}")
        print(f"✓ Time elapsed: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
        print(f"\nStatistics:")
        print(f"  Min: {metrics.min():.6f}")
        print(f"  Max: {metrics.max():.6f}")
        print(f"  Mean: {metrics.mean():.6f}")
        print(f"  Std: {metrics.std():.6f}")
        print(f"\nFirst 10 values: {metrics[:10]}")
        print("="*70)
        
        # Save a summary file
        summary_path = os.path.join(args.output_dir, f"{args.metric}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"ESD Metric Summary: {args.metric}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Computed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Time elapsed: {elapsed/60:.2f} minutes\n\n")
            f.write(f"Number of layers: {len(metrics)}\n")
            f.write(f"Min: {metrics.min():.6f}\n")
            f.write(f"Max: {metrics.max():.6f}\n")
            f.write(f"Mean: {metrics.mean():.6f}\n")
            f.write(f"Std: {metrics.std():.6f}\n")
        print(f"\nSummary saved to {summary_path}")
        
    except Exception as e:
        print(f"\n✗ Error computing metrics: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if model name is correct and accessible")
        print("  2. Ensure sufficient RAM/VRAM for model loading")
        print("  3. Verify model architecture is LLaMA/OPT-style")
        print("  4. Check dependencies are installed (weightwatcher, torch, transformers)")
        raise


if __name__ == '__main__':
    main()

