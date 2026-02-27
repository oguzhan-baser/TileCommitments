"""
Simple test script to demonstrate significance score extraction.
This script can be run interactively to explore the data.
"""

import numpy as np
import argparse
import sys

def test_metric_loading():
    """Test loading pre-computed metrics."""
    print("Testing metric loading...")
    
    # Check if metric files exist
    metric_path = "./data/llama-7b-hf/alpha_peak.npy"
    
    try:
        metrics = np.load(metric_path)
        print(f"✓ Successfully loaded metrics from {metric_path}")
        print(f"  Shape: {metrics.shape}")
        print(f"  Type: {type(metrics)}")
        print(f"  First 5 values: {metrics[:5]}")
        print(f"  Min: {metrics.min():.6f}, Max: {metrics.max():.6f}, Mean: {metrics.mean():.6f}")
        return metrics
    except FileNotFoundError:
        print(f"✗ Metric file not found: {metric_path}")
        print("  Available metric directories:")
        print("    - llama-7b-hf/")
        print("    - llama-13b-hf/")
        print("    - llama-30b-hf/")
        print("    - llama-65b-hf/")
        print("    - llama2-7b-hf/")
        print("    - llama2-13b-hf/")
        print("    - llama2-70b-hf/")
        return None


def simulate_pruning_ratios(metrics, sparsity_ratio=0.7, epsilon=0.3):
    """
    Simulate the pruning ratio computation without loading the full model.
    This demonstrates how significance scores can be extracted.
    """
    print("\n" + "="*60)
    print("Simulating Pruning Ratio Computation")
    print("="*60)
    
    scores = np.array(metrics)
    min_score = scores.min()
    max_score = scores.max()
    
    s1 = 1.0 - epsilon  # e.g., 0.7
    s2 = 1.0 + epsilon  # e.g., 1.3
    
    print(f"Parameters:")
    print(f"  Sparsity ratio: {sparsity_ratio}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Pruning ratio range: [{s1}, {s2}]")
    print(f"  Score range: [{min_score:.6f}, {max_score:.6f}]")
    
    # Linear mapping (normalize to [s1, s2])
    if max_score > min_score:
        normalized_scores = (scores - min_score) / (max_score - min_score)
        layerwise_pruning_ratios = normalized_scores * (s2 - s1) + s1
    else:
        layerwise_pruning_ratios = np.full_like(scores, (s1 + s2) / 2)
    
    # Scale to match target sparsity (simplified - assumes uniform param distribution)
    # In reality, this accounts for different layer sizes
    avg_ratio = layerwise_pruning_ratios.mean()
    scaler = sparsity_ratio / avg_ratio
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    
    # Clip to [0, 1]
    layerwise_pruning_ratios = np.clip(layerwise_pruning_ratios, 0, 1)
    
    print(f"\nPruning Ratios:")
    print(f"  Min: {layerwise_pruning_ratios.min():.6f}")
    print(f"  Max: {layerwise_pruning_ratios.max():.6f}")
    print(f"  Mean: {layerwise_pruning_ratios.mean():.6f}")
    print(f"  First 10: {layerwise_pruning_ratios[:10]}")
    
    return layerwise_pruning_ratios


def compute_significance_scores_demo(pruning_ratios, metrics):
    """Demonstrate different ways to compute significance scores."""
    print("\n" + "="*60)
    print("Computing Significance Scores (Multiple Methods)")
    print("="*60)
    
    # Method 1: Inverse ratio
    sig_inverse = 1.0 - pruning_ratios
    
    # Method 2: Normalized inverse
    sig_norm_inv = (sig_inverse - sig_inverse.min()) / (sig_inverse.max() - sig_inverse.min() + 1e-8)
    
    # Method 3: Based on raw metrics (inverse)
    metrics_norm = (metrics.max() - metrics) / (metrics.max() - metrics.min() + 1e-8)
    
    print("\nMethod 1: Inverse Pruning Ratio")
    print(f"  Range: [{sig_inverse.min():.6f}, {sig_inverse.max():.6f}]")
    print(f"  Mean: {sig_inverse.mean():.6f}")
    print(f"  First 5: {sig_inverse[:5]}")
    
    print("\nMethod 2: Normalized Inverse Ratio")
    print(f"  Range: [{sig_norm_inv.min():.6f}, {sig_norm_inv.max():.6f}]")
    print(f"  Mean: {sig_norm_inv.mean():.6f}")
    print(f"  First 5: {sig_norm_inv[:5]}")
    
    print("\nMethod 3: Based on Raw ESD Metrics")
    print(f"  Range: [{metrics_norm.min():.6f}, {metrics_norm.max():.6f}]")
    print(f"  Mean: {metrics_norm.mean():.6f}")
    print(f"  First 5: {metrics_norm[:5]}")
    
    # Show correlation
    print("\nCorrelations:")
    corr_1_2 = np.corrcoef(sig_inverse, sig_norm_inv)[0, 1]
    corr_1_3 = np.corrcoef(sig_inverse, metrics_norm)[0, 1]
    print(f"  Method 1 vs Method 2: {corr_1_2:.6f}")
    print(f"  Method 1 vs Method 3: {corr_1_3:.6f}")
    
    return {
        'inverse_ratio': sig_inverse,
        'normalized_inverse': sig_norm_inv,
        'metric_based': metrics_norm
    }


def show_top_layers(significance_scores, method_name, top_k=10):
    """Show top and bottom layers by significance."""
    print(f"\n" + "="*60)
    print(f"Top {top_k} Layers by Significance ({method_name})")
    print("="*60)
    
    sorted_indices = np.argsort(significance_scores)[::-1]  # Descending
    
    print(f"\nTop {top_k} Most Significant:")
    for i, idx in enumerate(sorted_indices[:top_k], 1):
        print(f"  {i:3d}. Layer {idx:3d}: {significance_scores[idx]:.6f}")
    
    print(f"\nBottom {top_k} Least Significant:")
    for i, idx in enumerate(sorted_indices[-top_k:], len(significance_scores)-top_k+1):
        print(f"  {i:3d}. Layer {idx:3d}: {significance_scores[idx]:.6f}")


def main():
    print("="*60)
    print("ALPHAPRUNING SIGNIFICANCE SCORE EXTRACTION - TEST")
    print("="*60)
    
    # Test metric loading
    metrics = test_metric_loading()
    if metrics is None:
        print("\nCannot proceed without metrics. Please check the data directory.")
        return
    
    # Simulate pruning ratio computation
    pruning_ratios = simulate_pruning_ratios(metrics, sparsity_ratio=0.7, epsilon=0.3)
    
    # Compute significance scores using different methods
    all_significance = compute_significance_scores_demo(pruning_ratios, metrics)
    
    # Show top layers for each method
    for method_name, scores in all_significance.items():
        show_top_layers(scores, method_name, top_k=10)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("  1. Significance scores can be extracted from pruning ratios")
    print("  2. Multiple methods yield similar relative rankings")
    print("  3. Lower pruning ratio → Higher significance")
    print("  4. ESD metrics themselves can be used as significance indicators")
    print("\nNext Steps:")
    print("  - Use extract_significance_scores.py for full pipeline integration")
    print("  - Specify desired method and output format")
    print("  - Map scores to actual layer names using find_layers()")


if __name__ == '__main__':
    main()

