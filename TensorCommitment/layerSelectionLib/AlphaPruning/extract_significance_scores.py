"""
Script to extract layer significance scores from AlphaPruning pipeline.

This script demonstrates different ways to extract significance scores
based on the pruning factors computed by AlphaPruning.
"""

import argparse
import os
import numpy as np
import torch
import json
from transformers import AutoModelForCausalLM

from lib.prune import find_layers
from lib.esd_utils import get_esd_metrics


def load_or_compute_metrics(args):
    """Load pre-computed metrics or compute them on-the-fly."""
    metric_path = f"{args.ww_metric_cache}/{args.ww_metric}.npy"
    
    if os.path.exists(metric_path):
        print(f"Loading metrics from {metric_path}")
        metrics = np.load(metric_path)
    else:
        print(f"Computing metrics for {args.model}...")
        metrics = get_esd_metrics(args.model, args.ww_metric, args.cache_dir)
        os.makedirs(args.ww_metric_cache, exist_ok=True)
        np.save(metric_path, metrics)
        print(f"Saved metrics to {metric_path}")
    
    return metrics


def compute_pruning_ratios(args, model, metrics):
    """
    Compute layer-wise pruning ratios using the same logic as ww_sparsity().
    Returns pruning ratios and intermediate values for analysis.
    """
    if "opt" in args.model.lower():
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    layers = find_layers(blocks)
    prunables = []
    layer_names = []
    
    for name in layers:
        prunables.append(layers[name].weight.numel())
        layer_names.append(name)
    
    layer_num_in_block = int(len(prunables) / len(blocks))
    
    # Handle block-wise mapping
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) 
                        for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    
    scores = torch.tensor(metrics)
    prunables_tensor = torch.tensor(prunables)
    
    # Linear mapping (same as ww_sparsity)
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    
    min_score = torch.min(scores)
    max_score = torch.max(scores)
    
    layerwise_pruning_ratios = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    
    # Scale to match target sparsity
    scaler = torch.sum(prunables_tensor) * args.sparsity_ratio / (torch.sum(prunables_tensor * layerwise_pruning_ratios))
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    
    return {
        'layerwise_pruning_ratios': layerwise_pruning_ratios,
        'raw_metrics': metrics,
        'normalized_scores': scores.cpu().numpy(),
        'layer_names': layer_names,
        'prunable_params': prunables,
        'min_score': min_score.item(),
        'max_score': max_score.item(),
        'min_pruning_ratio': min(layerwise_pruning_ratios),
        'max_pruning_ratio': max(layerwise_pruning_ratios)
    }


def compute_significance_scores(pruning_data, method='inverse_ratio'):
    """
    Compute significance scores from pruning ratios using different methods.
    
    Args:
        pruning_data: Dictionary containing pruning ratios and metrics
        method: Method to compute significance
            - 'inverse_ratio': 1 - pruning_ratio (higher = more significant)
            - 'normalized_inverse': Normalized inverse ratio [0, 1]
            - 'metric_based': Based on raw ESD metrics
            - 'inverse_metric': Inverse of normalized metric
    
    Returns:
        Dictionary mapping layer names to significance scores
    """
    ratios = pruning_data['layerwise_pruning_ratios']
    raw_metrics = pruning_data['raw_metrics']
    layer_names = pruning_data['layer_names']
    
    if method == 'inverse_ratio':
        # Simple inverse: less pruning = more significant
        scores = [1.0 - ratio for ratio in ratios]
        
    elif method == 'normalized_inverse':
        # Normalized inverse ratio
        inv_ratios = [1.0 - ratio for ratio in ratios]
        min_inv = min(inv_ratios)
        max_inv = max(inv_ratios)
        if max_inv > min_inv:
            scores = [(inv - min_inv) / (max_inv - min_inv) for inv in inv_ratios]
        else:
            scores = [0.5] * len(inv_ratios)
            
    elif method == 'metric_based':
        # Based on raw ESD metrics (inverse: lower metric = more significant)
        min_metric = min(raw_metrics)
        max_metric = max(raw_metrics)
        if max_metric > min_metric:
            scores = [(max_metric - m) / (max_metric - min_metric) for m in raw_metrics]
        else:
            scores = [0.5] * len(raw_metrics)
            
    elif method == 'inverse_metric':
        # Direct use of normalized metrics (higher metric = less significant)
        min_metric = min(raw_metrics)
        max_metric = max(raw_metrics)
        if max_metric > min_metric:
            scores = [(m - min_metric) / (max_metric - min_metric) for m in raw_metrics]
        else:
            scores = [0.5] * len(raw_metrics)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return dict(zip(layer_names, scores))


def print_analysis(pruning_data, significance_scores):
    """Print detailed analysis of significance scores."""
    print("\n" + "="*80)
    print("SIGNIFICANCE SCORE ANALYSIS")
    print("="*80)
    
    layer_names = pruning_data['layer_names']
    ratios = pruning_data['layerwise_pruning_ratios']
    raw_metrics = pruning_data['raw_metrics']
    
    # Sort by significance score (highest first)
    sorted_items = sorted(
        zip(layer_names, significance_scores.values(), ratios, raw_metrics),
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"\nTop 10 Most Significant Layers:")
    print(f"{'Rank':<6} {'Layer Name':<50} {'Significance':<15} {'Pruning Ratio':<15} {'ESD Metric':<15}")
    print("-" * 100)
    for rank, (name, sig_score, ratio, metric) in enumerate(sorted_items[:10], 1):
        print(f"{rank:<6} {name:<50} {sig_score:<15.6f} {ratio:<15.6f} {metric:<15.6f}")
    
    print(f"\nBottom 10 Least Significant Layers:")
    print(f"{'Rank':<6} {'Layer Name':<50} {'Significance':<15} {'Pruning Ratio':<15} {'ESD Metric':<15}")
    print("-" * 100)
    for rank, (name, sig_score, ratio, metric) in enumerate(sorted_items[-10:], len(sorted_items)-9):
        print(f"{rank:<6} {name:<50} {sig_score:<15.6f} {ratio:<15.6f} {metric:<15.6f}")
    
    print(f"\nStatistics:")
    print(f"  Total layers: {len(layer_names)}")
    print(f"  Significance range: [{min(significance_scores.values()):.6f}, {max(significance_scores.values()):.6f}]")
    print(f"  Pruning ratio range: [{pruning_data['min_pruning_ratio']:.6f}, {pruning_data['max_pruning_ratio']:.6f}]")
    print(f"  ESD metric range: [{pruning_data['min_score']:.6f}, {pruning_data['max_score']:.6f}]")


def save_significance_scores(significance_scores, pruning_data, output_path, format='json'):
    """Save significance scores to file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format == 'json':
        # Save as JSON with additional metadata
        output_data = {
            'significance_scores': significance_scores,
            'metadata': {
                'num_layers': len(significance_scores),
                'min_significance': min(significance_scores.values()),
                'max_significance': max(significance_scores.values()),
                'pruning_ratios': dict(zip(pruning_data['layer_names'], pruning_data['layerwise_pruning_ratios'])),
                'esd_metrics': dict(zip(pruning_data['layer_names'], pruning_data['raw_metrics']))
            }
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved significance scores to {output_path}")
        
    elif format == 'npy':
        # Save as numpy array (aligned with layer_names)
        layer_names = pruning_data['layer_names']
        scores_array = np.array([significance_scores[name] for name in layer_names])
        np.save(output_path, scores_array)
        print(f"\nSaved significance scores array to {output_path}")
        print(f"  Shape: {scores_array.shape}")
        print(f"  Aligned with layer_names in order")


def main():
    parser = argparse.ArgumentParser(description='Extract significance scores from AlphaPruning pipeline')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, help='Model identifier or path')
    parser.add_argument('--cache_dir', default="llm_weights", type=str, help='Model cache directory')
    
    # Metric arguments
    parser.add_argument('--ww_metric', default="alpha_peak", type=str, help='ESD metric to use')
    parser.add_argument('--ww_metric_cache', default="./data", type=str, help='Directory for metric cache')
    parser.add_argument('--mapping_type', default="block_wise", type=str, help='Mapping type')
    
    # Pruning arguments
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Target sparsity ratio')
    parser.add_argument('--epsilon', default=0.3, type=float, help='Epsilon for pruning ratio allocation')
    
    # Significance score arguments
    parser.add_argument('--method', default='inverse_ratio', type=str,
                       choices=['inverse_ratio', 'normalized_inverse', 'metric_based', 'inverse_metric'],
                       help='Method to compute significance scores')
    parser.add_argument('--output', type=str, default=None, help='Output path for significance scores')
    parser.add_argument('--format', default='json', choices=['json', 'npy'], help='Output format')
    parser.add_argument('--no_print', action='store_true', help='Skip printing analysis')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ALPHAPRUNING SIGNIFICANCE SCORE EXTRACTION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Metric: {args.ww_metric}")
    print(f"Sparsity Ratio: {args.sparsity_ratio}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Significance Method: {args.method}")
    
    # Load model (just to get structure, we don't need full weights)
    print("\nLoading model structure...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="cpu"  # Use CPU to avoid loading full weights
    )
    
    # Load or compute metrics
    metrics = load_or_compute_metrics(args)
    print(f"Loaded {len(metrics)} metric values")
    
    # Compute pruning ratios
    print("\nComputing layer-wise pruning ratios...")
    pruning_data = compute_pruning_ratios(args, model, metrics)
    print(f"Computed ratios for {len(pruning_data['layerwise_pruning_ratios'])} layers")
    print(f"Pruning ratio range: [{pruning_data['min_pruning_ratio']:.4f}, {pruning_data['max_pruning_ratio']:.4f}]")
    
    # Compute significance scores
    print(f"\nComputing significance scores using method: {args.method}...")
    significance_scores = compute_significance_scores(pruning_data, method=args.method)
    
    # Print analysis
    if not args.no_print:
        print_analysis(pruning_data, significance_scores)
    
    # Save if requested
    if args.output:
        save_significance_scores(significance_scores, pruning_data, args.output, format=args.format)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()

