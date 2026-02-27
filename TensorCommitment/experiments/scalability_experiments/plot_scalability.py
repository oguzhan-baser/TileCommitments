#!/usr/bin/env python3
"""
Plot scalability results with separate subplots for each model.

Shows how AlphaPrun benefit ratio changes with increasing number of verifiers.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Set seaborn darkgrid style
sns.set_style("darkgrid")


def load_results(results_file):
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def extract_scalability_data(results_data):
    """
    Extract scalability data organized by method and number of verifiers.
    
    Returns:
        Dictionary: {method: {num_verifiers: [ratios]}}
    """
    method_data = {}
    
    for case_result in results_data['results']:
        num_verifiers = case_result['num_verifiers']
        results = case_result['results']
        
        for method, method_result in results.items():
            if 'error' in method_result:
                continue
            
            # Get ratio (prefer alphaprun_ratio, fallback to alphaprun_benefit)
            if 'alphaprun_ratio' in method_result:
                ratio = method_result['alphaprun_ratio']
            elif 'alphaprun_benefit' in method_result:
                # Calculate ratio if we have total score
                total_score = results_data.get('total_alphaprun_score', 1.0)
                ratio = method_result['alphaprun_benefit'] / total_score if total_score > 0 else 0.0
            else:
                continue
            
            if method not in method_data:
                method_data[method] = {}
            
            if num_verifiers not in method_data[method]:
                method_data[method][num_verifiers] = []
            
            method_data[method][num_verifiers].append(ratio)
    
    return method_data


def plot_scalability(results_files, output_file):
    """
    Plot scalability results with separate subplots for each model.
    
    Args:
        results_files: List of paths to results JSON files (one per model)
        output_file: Path to save plot
    """
    methods = ['random', 'norm_inf', 'norm_2', 'norm_1', 'alphaprun']
    method_names = {
        'random': 'Random',
        'norm_inf': 'WeightNorm L∞',
        'norm_2': 'WeightNorm L2',
        'norm_1': 'WeightNorm L1',
        'alphaprun': 'AlphaPrun'
    }
    
    # Color mapping (same as previous experiments)
    method_colors = {
        'random': '#FF6B6B',      # Red
        'norm_inf': '#4ECDC4',    # Cyan
        'norm_2': '#45B7D1',      # Blue
        'norm_1': '#FFA07A',      # Salmon
        'alphaprun': '#6C5CE7'    # Purple
    }
    
    # Load data for each model
    model_data = {}
    model_names = []
    
    for results_file in results_files:
        results_data = load_results(results_file)
        model_name = results_data.get('model_name', Path(results_file).stem)
        model_names.append(model_name)
        
        method_data = extract_scalability_data(results_data)
        model_data[model_name] = method_data
    
    if len(model_data) == 0:
        print("No data to plot!")
        return
    
    # Create subplots: one per model
    num_models = len(model_names)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    
    # Collect all legend handles and labels (for shared legend)
    all_handles = []
    all_labels = []
    
    # Plot each model
    for model_idx, model_name in enumerate(model_names):
        ax = axes[model_idx]
        method_data = model_data[model_name]
        
        # Collect data for each method
        for method in methods:
            if method not in method_data:
                continue
            
            # Get all verifier counts and their ratios, filter up to 81
            verifier_counts = sorted([vc for vc in method_data[method].keys() if vc <= 81])
            if len(verifier_counts) == 0:
                continue
            
            means = []
            stds = []
            
            for num_verifiers in verifier_counts:
                # Calculate mean and std for multiple cases per verifier count
                ratios_for_count = method_data[method][num_verifiers]
                mean_ratio = np.mean(ratios_for_count)
                std_ratio = np.std(ratios_for_count)
                means.append(mean_ratio)
                stds.append(std_ratio)
            
            # Plot mean line with shaded std region
            color = method_colors.get(method, '#000000')
            line = ax.plot(verifier_counts, means, 
                   marker='o', markersize=4, linewidth=2,
                   label=method_names[method], color=color, alpha=0.8)
            
            # Add shaded region for standard deviation
            upper_bound = [m + s for m, s in zip(means, stds)]
            lower_bound = [m - s for m, s in zip(means, stds)]
            ax.fill_between(verifier_counts, lower_bound, upper_bound,
                          color=color, alpha=0.2, linewidth=0)
            
            # Collect handles and labels for shared legend (only once per method)
            if model_idx == 0:  # Only collect from first subplot
                all_handles.append(line[0])
                all_labels.append(method_names[method])
        
        # Customize subplot
        ax.set_xlabel('Number of Verifiers', fontsize=12, fontweight='bold')
        ax.set_ylabel('AMC', fontsize=12, fontweight='bold')
        ax.set_title(model_name, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits and ticks
        ax.set_xlim(left=0, right=85)  # Slightly beyond 81 for better visualization
        if len(verifier_counts) > 0:
            # Show every other tick to avoid crowding
            tick_positions = verifier_counts[::2]
            ax.set_xticks(tick_positions)
    
    # Remove extra subplots if we have fewer than 4 models
    for idx in range(num_models, 4):
        fig.delaxes(axes[idx])
    
    # Overall title
    fig.suptitle('Scalability: AMC vs Number of Verifiers\n'
                 '(Each verifier gets 1% of total parameters)',
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Create shared legend at the top, close to title
    if len(all_handles) > 0:
        fig.legend(handles=all_handles, labels=all_labels,
                  loc='upper center', bbox_to_anchor=(0.5, 0.94),
                  ncol=len(all_handles), fontsize=12, framealpha=0.95,
                  title='Optimization Method', title_fontsize=13)
        # Make legend title bold
        legend = fig.legends[0]
        legend.get_title().set_fontweight('bold')
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.subplots_adjust(hspace=0.125 , wspace=0.025) 
    axes[1].yaxis.set_visible(False)
    axes[3].yaxis.set_visible(False)
    axes[0].xaxis.set_visible(False)
    axes[1].xaxis.set_visible(False)
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Plot scalability results'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file(s) - comma-separated or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output plot file path')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    
    # Handle multiple input formats
    if results_path.is_dir():
        # Find all scalability results files
        results_files = list(results_path.glob("*_scalability_results.json"))
    elif ',' in args.results:
        # Comma-separated list
        results_files = [Path(f.strip()) for f in args.results.split(',')]
    else:
        # Single file
        results_files = [results_path]
    
    if len(results_files) == 0:
        print(f"No results files found")
        return
    
    print(f"Plotting scalability results from {len(results_files)} file(s)...")
    plot_scalability(results_files, args.output)


if __name__ == '__main__':
    main()

