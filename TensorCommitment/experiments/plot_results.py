#!/usr/bin/env python3
"""
Plot boxplots comparing different optimization methods.

Creates boxplots showing the distribution of alphaprun benefits
achieved by each optimization method across all cases.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn darkgrid style
sns.set_style("darkgrid")


def load_results(results_file):
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def extract_benefits_by_method(results_data, use_ratio=True):
    """
    Extract alphaprun benefits (or ratios) for each method across all cases.
    
    Args:
        results_data: Results data dictionary
        use_ratio: If True, use alphaprun_ratio (default). If False, use raw alphaprun_benefit.
    
    Returns:
        Dictionary mapping method name to list of benefits/ratios
    """
    method_benefits = {}
    
    for case_result in results_data['results']:
        case_id = case_result['case_id']
        results = case_result['results']
        
        for method, method_result in results.items():
            if 'error' in method_result:
                continue
            
            # Prefer ratio if available and use_ratio is True, otherwise use raw benefit
            if use_ratio and 'alphaprun_ratio' in method_result:
                value = method_result['alphaprun_ratio']
            elif 'alphaprun_benefit' in method_result:
                value = method_result['alphaprun_benefit']
            else:
                continue
            
            if method not in method_benefits:
                method_benefits[method] = []
            method_benefits[method].append(value)
    
    return method_benefits


def plot_boxplots(method_benefits, output_file, model_name=None):
    """
    Create boxplot comparing methods.
    
    Args:
        method_benefits: Dictionary mapping method to list of benefits
        output_file: Path to save plot
        model_name: Model name for title
    """
    # Method display names
    method_names = {
        'random': 'Random',
        'norm_1': 'WeightNorm L1',
        'norm_2': 'WeightNorm L2',
        'norm_inf': 'WeightNorm L∞',
        'alphaprun': 'AlphaPrun'
    }
    
    # Prepare data for plotting
    data_for_plot = []
    method_labels = []
    
    for method, benefits in method_benefits.items():
        if len(benefits) == 0:
            continue
        data_for_plot.append(benefits)
        method_labels.append(method_names.get(method, method))
    
    if len(data_for_plot) == 0:
        print("No data to plot!")
        return
    
    # Create figure with darkgrid style
    plt.figure(figsize=(12, 7))
    
    # Create boxplot
    bp = plt.boxplot(data_for_plot, tick_labels=method_labels, patch_artist=True)
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#6C5CE7']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # Customize plot
    plt.ylabel('AlphaPrun Benefit Ratio', fontsize=13, fontweight='bold')
    plt.xlabel('Optimization Method', fontsize=13, fontweight='bold')
    plt.title(f'Comparison of Optimization Methods{" - " + model_name if model_name else ""}',
              fontsize=15, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    
    # Add statistics text
    stats_text = []
    for i, (method, benefits) in enumerate(method_benefits.items()):
        if len(benefits) > 0:
            mean_val = np.mean(benefits)
            median_val = np.median(benefits)
            stats_text.append(f"{method_names.get(method, method)}: "
                            f"mean={mean_val:.2f}, median={median_val:.2f}")
    
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also save statistics
    stats_file = output_path.with_suffix('.txt')
    with open(stats_file, 'w') as f:
        f.write("Optimization Method Statistics\n")
        f.write("=" * 50 + "\n\n")
        for line in stats_text:
            f.write(line + "\n")
    print(f"Statistics saved to {stats_file}")
    
    plt.close()


def plot_combined_results(results_files, output_file):
    """
    Plot combined results from multiple models side by side.
    X-axis: Models (datasets)
    Colors/Legend: Optimization methods
    
    Args:
        results_files: List of paths to results JSON files
        output_file: Path to save combined plot
    """
    methods = ['random', 'norm_inf', 'norm_2', 'norm_1', 'alphaprun']
    method_names = {
        'random': 'Random',
        'norm_inf': 'WeightNorm L∞',
        'norm_2': 'WeightNorm L2',
        'norm_1': 'WeightNorm L1',
        'alphaprun': 'AlphaPrun'
    }
    
    # Collect data organized by model, then by method
    model_data = {}  # {model_name: {method: [benefits]}}
    model_names = []
    
    for results_file in results_files:
        results_data = load_results(results_file)
        model_name = results_data.get('model_name', Path(results_file).stem)
        model_names.append(model_name)
        
        # Use ratio by default (True)
        method_benefits = extract_benefits_by_method(results_data, use_ratio=True)
        model_data[model_name] = method_benefits
    
    if len(model_data) == 0:
        print("No data to plot!")
        return
    
    # Prepare data for grouped boxplot
    # Structure: For each model, we have data for each method
    num_models = len(model_names)
    num_methods = len(methods)
    method_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#6C5CE7']  # 5 colors for 5 methods
    
    # Calculate positions for grouped boxes
    # Each model group will have boxes for each method, spaced evenly
    group_width = num_methods + 0.5  # Width of each model group
    base_positions = np.arange(1, num_models + 1) * group_width
    
    # Create data structure for grouped boxplot
    all_plot_data = []  # List of lists, one per (model, method) combination
    all_positions = []  # X positions for each box
    all_colors = []  # Colors for each box
    
    for model_idx, model_name in enumerate(model_names):
        model_base_pos = base_positions[model_idx]
        
        for method_idx, method in enumerate(methods):
            if method in model_data[model_name] and len(model_data[model_name][method]) > 0:
                # Position: base position + method offset
                # Space methods evenly within each group
                offset = (method_idx - (num_methods - 1) / 2) * 0.8
                position = model_base_pos + offset
                
                all_plot_data.append(model_data[model_name][method])
                all_positions.append(position)
                all_colors.append(method_colors[method_idx])
    
    if len(all_plot_data) == 0:
        print("No data to plot!")
        return
    
    # Create a single figure (not subplots)
    plt.figure(figsize=(12.5, 6))
    ax = plt.gca()
    # Create boxplot with custom positions
    bp = ax.boxplot(all_plot_data, positions=all_positions, widths=0.5, 
                    patch_artist=True, showfliers=False)
    
    # Color the boxes according to method
    for i, (patch, color) in enumerate(zip(bp['boxes'], all_colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # Set x-axis labels (model names) at group centers
    ax.set_xticks(base_positions)
    ax.set_xticklabels(model_names, fontsize=13, fontweight='bold', rotation=0)
    
    # Add subtle vertical lines to separate model groups for clarity
    for pos in base_positions:
        ax.axvline(x=pos - group_width/2, color='gray', linestyle='--', 
                  alpha=0.2, linewidth=0.8)
    
    # Create clear legend for methods with all colors
    legend_elements = []
    for i, method in enumerate(methods):
        legend_elements.append(
            plt.Rectangle((0,0), 1, 1, 
                         facecolor=method_colors[i], 
                         alpha=0.7, 
                         edgecolor='black', 
                         linewidth=1.2,
                         label=method_names[method])
        )
    
    # Customize plot
    ax.set_ylabel('AlphaPrun Benefit Ratio', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    
    # Set title first with extra padding at top for legend
    ax.set_title('Comparison of Optimization Methods Across Models', 
                 fontsize=16, fontweight='bold', pad=60)  # Extra padding for legend below
    
    # Place legend horizontally just below the title
    # ncol = number of methods (6) for horizontal layout: 1 row × 6 columns
    legend = ax.legend(handles=legend_elements, 
                       loc='upper center', 
                       bbox_to_anchor=(0.5, 1.2),  # Position just below title
                       ncol=num_methods,  # Horizontal: 1 row, num_methods columns
                       fontsize=13,  # Increased font size for method names
                       framealpha=0.95,
                       title='Optimization Method', 
                       title_fontsize=13,
                       columnspacing=1.5,
                       handlelength=1.5,
                       handletextpad=0.5)
    # Make legend title bold
    legend.get_title().set_fontweight('bold')
    
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.8)
    
    # Adjust xlim to remove unreasonable left padding
    # Start closer to the first group, end with some padding on right
    left_margin = base_positions[0] - group_width/2 - 0.5
    right_margin = max(base_positions) + group_width/2 + 0.5
    ax.set_xlim(left=left_margin, right=right_margin)
    
    # Add statistics text
    stats_text = []
    for model_name in model_names:
        stats_text.append(f"\n{model_name}:")
        for method in methods:
            if method in model_data[model_name] and len(model_data[model_name][method]) > 0:
                benefits = model_data[model_name][method]
                mean_val = np.mean(benefits)
                median_val = np.median(benefits)
                stats_text.append(f"  {method_names[method]}: mean={mean_val:.2f}, median={median_val:.2f}")
    
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.ylabel('ACM', fontsize=13, fontweight='bold')
    if not os.path.exists(output_path):
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_path}")
    
    # Save statistics
    stats_file = output_path.with_suffix('.txt')
    with open(stats_file, 'w') as f:
        f.write("Optimization Method Statistics by Model\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Models included: {', '.join(model_names)}\n")
        f.write(f"Total cases per model: {len(model_data[model_names[0]].get('alphaprun', []))} cases\n\n")
        for line in stats_text:
            f.write(line + "\n")
    print(f"Statistics saved to {stats_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file (or directory with multiple)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output plot file path')
    parser.add_argument('--combined', action='store_true',
                       help='Combine results from multiple models')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    
    if args.combined or results_path.is_dir():
        # Find all results files
        if results_path.is_dir():
            results_files = list(results_path.glob("*_all_results.json"))
        else:
            # Assume comma-separated list
            results_files = [Path(f.strip()) for f in args.results.split(',')]
        
        if len(results_files) == 0:
            print(f"No results files found in {results_path}")
            return
        
        plot_combined_results(results_files, args.output)
    else:
        # Single results file
        results_data = load_results(results_path)
        model_name = results_data.get('model_name', None)
        # Use ratio by default (True)
        method_benefits = extract_benefits_by_method(results_data, use_ratio=True)
        plot_boxplots(method_benefits, args.output, model_name)


if __name__ == '__main__':
    main()

