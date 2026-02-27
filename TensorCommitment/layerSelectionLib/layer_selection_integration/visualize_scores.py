"""
Visualization script for layer significance scores.

Creates bar plots showing significance scores for each layer index,
with support for comparing multiple models in a single plot.
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set seaborn dark grid theme
sns.set_style("darkgrid")

# Add AlphaPruning to path
ALPHAPRUNING_PATH = Path(__file__).parent.parent / "AlphaPruning"
sys.path.insert(0, str(ALPHAPRUNING_PATH))


def find_metric_files(data_dir, metric_name):
    """Find all .npy metric files in the data directory."""
    metric_files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        return metric_files
    
    # Look for metric files in subdirectories
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir():
            metric_file = subdir / f"{metric_name}.npy"
            if metric_file.exists():
                metric_files.append({
                    'path': metric_file,
                    'model_name': subdir.name,
                    'full_path': str(metric_file)
                })
    
    return metric_files


def load_json_scores(json_file):
    """Load significance scores from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract scores in order
    layers = sorted(data['layers'], key=lambda x: x['index'])
    indices = [l['index'] for l in layers]
    scores = [l['significance_score'] for l in layers]
    model_name = data.get('model_name', Path(json_file).stem)
    
    return {
        'indices': indices,
        'scores': scores,
        'model_name': model_name
    }


def load_metrics_from_npy(npy_file, model_name, normalize=True, use_raw_metrics=False):
    """
    Load metrics from .npy file and compute significance scores.
    
    Args:
        npy_file: Path to .npy file
        model_name: Name of the model
        normalize: If True, normalize to [0,1]. If False, return raw inverse metrics.
        use_raw_metrics: If True, use raw ESD metrics directly (inverted) without any transformation
    """
    metrics = np.load(npy_file)
    
    if use_raw_metrics:
        # Use raw ESD metrics directly, just invert for significance (lower alpha = higher significance)
        # This shows the actual raw metric differences
        significance_scores = max(metrics) - metrics  # Simple inverse: higher alpha -> lower significance
    elif normalize:
        # For visualization, normalize to [0,1] to show significance (lower metric = higher significance)
        min_metric = metrics.min()
        max_metric = metrics.max()
        
        if max_metric > min_metric:
            # Inverse: lower metric = higher significance, normalized to [0,1]
            significance_scores = (max_metric - metrics) / (max_metric - min_metric)
        else:
            significance_scores = np.ones_like(metrics) * 0.5
    else:
        # Raw inverse metrics (not normalized): lower metric = higher significance
        # This preserves the actual scale differences but inverts for significance
        significance_scores = max(metrics) - metrics  # Inverse, but not normalized
    
    return {
        'indices': list(range(len(metrics))),
        'scores': significance_scores.tolist(),
        'model_name': model_name,
        'raw_metrics': metrics.tolist()
    }


def plot_single_model(indices, scores, model_name, ax=None, color=None, alpha=0.7):
    """Plot significance scores for a single model."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(indices, scores, alpha=alpha, label=model_name, color=color)
    return ax


def plot_multiple_models_from_data_dir(data_dir, metric_name, output_file, 
                                       max_models=None, figsize=(16, 8), normalize=True, use_raw_metrics=False):
    """
    Plot all models from data directory using quarter-based boxplots in a single plot.
    
    Creates one plot with boxplots grouped by quarters (Q1, Q2, Q3, Q4) with spacing
    between quarters. Model names appear in legend.
    """
    metric_files = find_metric_files(data_dir, metric_name)
    
    if not metric_files:
        print(f"No metric files found for '{metric_name}' in {data_dir}")
        return
    
    if max_models:
        metric_files = metric_files[:max_models]
    
    print(f"Found {len(metric_files)} models to plot:")
    for mf in metric_files:
        print(f"  - {mf['model_name']}")
    
    # Load all model data
    model_data = []
    for mf in metric_files:
        try:
            data = load_metrics_from_npy(mf['full_path'], mf['model_name'], normalize=normalize, use_raw_metrics=use_raw_metrics)
            model_data.append({
                'name': mf['model_name'],
                'scores': data['scores'],
                'indices': data['indices']
            })
        except Exception as e:
            print(f"Warning: Failed to load {mf['model_name']}: {e}")
            continue
    
    if not model_data:
        print("No models loaded successfully")
        return
    
    # Create single figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
    
    # Prepare data: collect all quarters for all models
    num_models = len(model_data)
    num_quarters = 4
    
    # Position calculation: groups of boxplots with spacing
    # Spacing: 1 gap between Q1-Q2, 2 gaps between Q2-Q3, 1 gap between Q3-Q4
    box_width = 0.8
    gap_between_models = 0.1
    gap_between_quarters = 1.5  # Space between quarter groups
    
    all_data = []
    all_positions = []
    all_colors = []
    legend_handles = []  # For legend
    
    current_position = 1
    
    for quarter_idx in range(num_quarters):
        quarter_start_pos = current_position
        
        # Collect scores for this quarter from all models
        for model_idx, model in enumerate(model_data):
            scores = np.array(model['scores'])
            n = len(scores)
            
            # Calculate quarter boundaries
            start_idx = int(quarter_idx * n / 4)
            end_idx = int((quarter_idx + 1) * n / 4)
            
            # Get scores for this quarter
            quarter_scores = scores[start_idx:end_idx]
            
            if len(quarter_scores) > 0:
                all_data.append(quarter_scores)
                all_positions.append(current_position)
                all_colors.append(colors[model_idx])
                
                # Create legend handle (only once per model)
                if quarter_idx == 0:
                    from matplotlib.patches import Rectangle
                    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=colors[model_idx], 
                                                    alpha=0.7, label=model['name']))
            
            # Move to next position for this model
            current_position += 1 + gap_between_models
        
        # Add spacing after quarter (except after Q4)
        if quarter_idx < num_quarters - 1:
            if quarter_idx == 1:  # After Q2, add extra spacing
                current_position += gap_between_quarters * 2
            else:
                current_position += gap_between_quarters
    
    # Create all boxplots
    bp = ax.boxplot(all_data, positions=all_positions, widths=box_width, patch_artist=True)
    
    # Color each boxplot
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the boxplot elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1)
    
    # Set x-axis labels and ticks
    # Place quarter labels at the center of each quarter group
    quarter_centers = []
    quarter_labels = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
    
    current_pos = 1
    for quarter_idx in range(num_quarters):
        quarter_start = current_pos
        # Calculate center of this quarter's boxplots
        quarter_center = quarter_start + (num_models - 1) * (1 + gap_between_models) / 2
        quarter_centers.append(quarter_center)
        
        # Move to start of next quarter
        current_pos += num_models * (1 + gap_between_models)
        if quarter_idx < num_quarters - 1:
            if quarter_idx == 1:
                current_pos += gap_between_quarters * 2
            else:
                current_pos += gap_between_quarters
    
    # Set x-axis ticks and labels (bigger font)
    ax.set_xticks(quarter_centers)
    ax.set_xticklabels(quarter_labels, fontsize=14, fontweight='bold')
    
    # Add vertical lines to separate quarters
    current_pos = 1
    for quarter_idx in range(num_quarters - 1):
        current_pos += num_models * (1 + gap_between_models)
        if quarter_idx == 1:
            current_pos += gap_between_quarters * 2
        else:
            current_pos += gap_between_quarters
        ax.axvline(x=current_pos - gap_between_quarters , color='gray', 
                   linestyle='--', linewidth=1, alpha=0.3)
    
    # Set labels (adjust based on normalization and raw metrics)
    if use_raw_metrics:
        ylabel = 'Significance Score (Raw ESD Metrics, Inverted)'
    elif normalize:
        ylabel = 'Significance Score (Normalized)'
    else:
        ylabel = 'Significance Score (Raw)'
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_xlabel('Quarter', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add title above everything
    fig.suptitle(f'Layer Significance Scores Comparison by Quarters\n(Metric: {metric_name})', 
                 fontsize=18, fontweight='bold', y=0.75)
    
    # Add legend below title, multiline (4 models per row), ordered row-wise
    ncol_legend = 4  # 4 models per row
    
    # Reorder handles for row-wise display (matplotlib fills column-wise by default)
    # To get row-wise: [1,2,3,4,5,6,7,8] displayed as:
    #   1  2  3  4
    #   5  6  7  8
    # We need to reorder to: [1,5,2,6,3,7,4,8] for matplotlib to display it row-wise
    num_models = len(legend_handles)
    num_rows = (num_models + ncol_legend - 1) // ncol_legend  # Ceiling division
    reordered_handles = []
    # Fill column by column (so matplotlib displays them row by row)
    for col in range(ncol_legend):
        for row in range(num_rows):
            idx = row * ncol_legend + col
            if idx < num_models:
                reordered_handles.append(legend_handles[idx])
    
    legend = ax.legend(handles=reordered_handles, loc='upper center', 
                       bbox_to_anchor=(0.5, 1.35), ncol=ncol_legend, 
                       fontsize=22, framealpha=0.9, frameon=True, 
                       columnspacing=1.5, handlelength=2.0, handletextpad=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space at top for title and legend
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    plt.close()


def plot_from_json_files(json_files, output_file, figsize=(16, 8)):
    """Plot from JSON files (from extract_layer_scores.py output)."""
    if isinstance(json_files, str):
        json_files = [json_files]
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(json_files)))
    
    for idx, json_file in enumerate(json_files):
        try:
            data = load_json_scores(json_file)
            indices = data['indices']
            scores = data['scores']
            
            width = 0.8 / len(json_files)
            offset = (idx - len(json_files) / 2 + 0.5) * width
            
            ax.bar(np.array(indices) + offset, scores, width=width,
                   alpha=0.7, label=data['model_name'], color=colors[idx])
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Significance Score (Raw)', fontsize=12)
    ax.set_title('Layer Significance Scores Comparison', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    plt.close()


def plot_single_from_json(json_file, output_file, figsize=(12, 6)):
    """Plot a single model from JSON file."""
    data = load_json_scores(json_file)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(data['indices'], data['scores'], alpha=0.7, color='steelblue')
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Significance Score (Raw)', fontsize=12)
    ax.set_title(f'Layer Significance Scores: {data["model_name"]}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize layer significance scores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mode', type=str, default='data_dir',
                       choices=['data_dir', 'json', 'json_single'],
                       help='Visualization mode')
    parser.add_argument('--data_dir', type=str,
                       default=str(ALPHAPRUNING_PATH / 'data'),
                       help='Directory containing model metric subdirectories')
    parser.add_argument('--metric', type=str, default='alpha_peak',
                       help='Metric name to visualize (for data_dir mode)')
    parser.add_argument('--json_file', type=str, nargs='+',
                       help='JSON file(s) from extract_layer_scores.py')
    parser.add_argument('--output', type=str, required=True,
                       help='Output plot file path')
    parser.add_argument('--max_models', type=int, default=None,
                       help='Maximum number of models to plot (for data_dir mode)')
    parser.add_argument('--figsize', type=str, default='16,8',
                       help='Figure size as "width,height"')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize scores to [0,1] range (default: True)')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false',
                       help='Use raw significance scores without normalization')
    parser.add_argument('--use_raw_metrics', action='store_true', default=False,
                       help='Use raw ESD metrics directly (inverted for significance) without computing pruning ratios')
    
    args = parser.parse_args()
    
    # Parse figure size
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except:
        figsize = (16, 8)
    
    print("="*70)
    print("LAYER SIGNIFICANCE SCORES VISUALIZATION")
    print("="*70)
    
    if args.mode == 'data_dir':
        print(f"Mode: Plot all models from data directory")
        print(f"Data directory: {args.data_dir}")
        print(f"Metric: {args.metric}")
        plot_multiple_models_from_data_dir(
            args.data_dir, args.metric, args.output,
            max_models=args.max_models, figsize=figsize, 
            normalize=args.normalize, use_raw_metrics=args.use_raw_metrics
        )
    
    elif args.mode == 'json':
        print(f"Mode: Plot from JSON files")
        if not args.json_file:
            print("Error: --json_file required for json mode")
            sys.exit(1)
        plot_from_json_files(args.json_file, args.output, figsize=figsize)
    
    elif args.mode == 'json_single':
        print(f"Mode: Plot single model from JSON file")
        if not args.json_file or len(args.json_file) != 1:
            print("Error: Exactly one --json_file required for json_single mode")
            sys.exit(1)
        plot_single_from_json(args.json_file[0], args.output, figsize=figsize)
    
    print("="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()

