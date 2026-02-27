#!/usr/bin/env python3
"""
Plot script to visualize dimension comparison benchmark results.
Creates line plots using seaborn with darkgrid style, showing how different
dimensional configurations (2D, 3D, 6D) perform at different data sizes.
"""
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="results/dimension_benchmark.jsonl",
        help="Input file with benchmark results (JSONL format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dimension_benchmark_plot.png",
        help="Output plot file",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output plot")
    parser.add_argument(
        "--base-size",
        type=int,
        default=64,
        help="Base size for x-axis formatting (e.g., 64 for 64^exp format)",
    )
    return parser.parse_args()


def load_benchmark_data(filename: str) -> pd.DataFrame:
    """Load benchmark results from JSONL file into a pandas DataFrame.
    Normalizes both lagrange_multiverkle and pegasus formats to a common format with seconds.
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue

    if not data:
        raise ValueError(f"No valid data found in {filename}")

    df = pd.DataFrame(data)
    
    # Normalize to common format: all times in seconds, unified column names
    # Check if we have pegasus format (has 'implementation' field)
    has_pegasus = "implementation" in df.columns
    has_lagrange = "config_name" in df.columns
    
    if not has_pegasus and not has_lagrange:
        raise ValueError("Data must have either 'implementation' (pegasus) or 'config_name' (lagrange) field")
    
    # Normalize size column - handle mixed data row by row
    def normalize_size(row):
        """Get size from either data_size or size column."""
        if pd.notna(row.get("data_size")):
            return row["data_size"]
        elif pd.notna(row.get("size")):
            return row["size"]
        else:
            return None
    
    # Create unified data_size column
    df["data_size"] = df.apply(normalize_size, axis=1)
    
    # Ensure data_size is numeric and filter out rows with NaN
    df["data_size"] = pd.to_numeric(df["data_size"], errors='coerce')
    initial_len = len(df)
    df = df[df["data_size"].notna()]
    if len(df) < initial_len:
        print(f"Warning: Filtered out {initial_len - len(df)} rows with invalid data_size")
    
    # Normalize times to seconds - handle mixed formats row by row
    def normalize_build_time(row):
        """Get build time in seconds from any available format."""
        if pd.notna(row.get("build_time_seconds")):
            return row["build_time_seconds"]
        elif pd.notna(row.get("build_time_ms")):
            return row["build_time_ms"] / 1000.0
        elif pd.notna(row.get("time_taken_to_build_tree")):
            return row["time_taken_to_build_tree"]
        else:
            return None
    
    def normalize_proof_time(row):
        """Get proof time in seconds from any available format."""
        if pd.notna(row.get("proof_time_seconds")):
            return row["proof_time_seconds"]
        elif pd.notna(row.get("proof_time_us")):
            return row["proof_time_us"] / 1_000_000.0
        elif pd.notna(row.get("time_taken_to_generate_proof")):
            return row["time_taken_to_generate_proof"]
        else:
            return 0.0
    
    def normalize_verify_time(row):
        """Get verify time in seconds from any available format."""
        if pd.notna(row.get("verify_time_seconds")):
            return row["verify_time_seconds"]
        elif pd.notna(row.get("verify_time_us")):
            return row["verify_time_us"] / 1_000_000.0
        elif pd.notna(row.get("time_taken_to_verify_proof")):
            return row["time_taken_to_verify_proof"]
        else:
            return 0.0
    
    # Apply normalization row by row
    df["build_time_seconds"] = df.apply(normalize_build_time, axis=1)
    df["proof_time_seconds"] = df.apply(normalize_proof_time, axis=1)
    df["verify_time_seconds"] = df.apply(normalize_verify_time, axis=1)
    
    # Check if we have at least some build times
    if df["build_time_seconds"].isna().all():
        raise ValueError("No build time data found in any format")
    
    # Create unified config_name/display_label - handle mixed formats row by row
    def create_display_label(row):
        """Create display_label based on what fields are available in this row."""
        if pd.notna(row.get("config_name")):
            return str(row["config_name"]), "lagrange_multiverkle"
        elif pd.notna(row.get("implementation")):
            impl = str(row["implementation"])
            if pd.notna(row.get("arity_label")):
                arity = str(row["arity_label"])
                return f"{impl}-{arity}", impl
            else:
                return impl, impl
        else:
            return "unknown", "unknown"
    
    # Apply row by row to handle mixed data
    display_impl = df.apply(create_display_label, axis=1)
    df["display_label"] = [x[0] for x in display_impl]
    df["implementation"] = [x[1] for x in display_impl]
    
    # Ensure display_label is string and fill any remaining NaN
    df["display_label"] = df["display_label"].fillna("unknown").astype(str)
    df["implementation"] = df["implementation"].fillna("unknown").astype(str)
    
    # Filter out any rows with invalid display_label
    df = df[df["display_label"] != "nan"]
    df = df[df["display_label"].notna()]
    
    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual runs by (display_label, data_size), computing mean and std."""
    # Debug: show what we're aggregating
    print(f"  Aggregating {len(df)} rows")
    print(f"  Unique display_labels: {df['display_label'].unique()}")
    print(f"  Unique data_sizes: {sorted(df['data_size'].dropna().unique())}")
    
    # Build agg_dict with normalized seconds columns
    # Always include all time columns if they exist, even if some groups have NaN
    agg_dict = {}
    
    if "build_time_seconds" in df.columns:
        agg_dict["build_time_seconds"] = ["mean", "std"]
    
    if "proof_time_seconds" in df.columns:
        agg_dict["proof_time_seconds"] = ["mean", "std"]
    
    if "verify_time_seconds" in df.columns:
        agg_dict["verify_time_seconds"] = ["mean", "std"]
    
    if not agg_dict:
        raise ValueError("No time columns found for aggregation")
    
    # Preserve implementation for grouping/coloring
    if "implementation" in df.columns:
        agg_dict["implementation"] = "first"  # Keep first value for grouping

    # Group and aggregate - dropna=False to keep groups even if some values are NaN
    grouped = df.groupby(["display_label", "data_size"], as_index=False, dropna=False).agg(agg_dict)
    
    # Flatten column names (pandas creates MultiIndex columns after aggregation)
    # All columns become tuples: ('config_name', ''), ('build_time_ms', 'mean'), etc.
    new_columns = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            metric = col[0]
            stat = col[1] if len(col) > 1 else ""
            
            # Grouping columns have empty second element: ('config_name', '')
            if stat == "":
                new_columns.append(metric)
            else:
                # Aggregated columns: ('build_time_ms', 'mean') -> 'build_time_ms_mean'
                new_columns.append(f"{metric}_{stat}")
        else:
            # Fallback (shouldn't happen with pandas MultiIndex)
            new_columns.append(str(col))
    
    grouped.columns = new_columns
    
    # Verify we still have the grouping columns
    if "display_label" not in grouped.columns:
        print(f"ERROR: display_label not found! Columns are: {list(grouped.columns)}")
        raise ValueError("display_label column was lost during aggregation!")
    if "data_size" not in grouped.columns:
        print(f"ERROR: data_size not found! Columns are: {list(grouped.columns)}")
        raise ValueError("data_size column was lost during aggregation!")
    
    return grouped


def format_size_label(size: int, base: int = 64) -> str:
    """Format size label as base^exp if perfect power, otherwise as number."""
    if size <= 0:
        return str(size)
    
    # Check if it's a perfect power of base_size
    exp = math.log(size) / math.log(base)
    if abs(exp - round(exp)) < 1e-6:  # Close to integer
        return f"{base}^{int(round(exp))}"
    else:
        # Format large numbers nicely
        if size >= 1e6:
            return f"{size/1e6:.1f}M"
        elif size >= 1e3:
            return f"{size/1e3:.1f}K"
        else:
            return str(size)


def main() -> None:
    args = parse_args()
    
    # Load data
    print(f"Loading benchmark data from {args.input}...")
    df = load_benchmark_data(args.input)
    print(f"Loaded {len(df)} benchmark results")
    
    # Show what display labels were created
    if "display_label" in df.columns:
        label_counts = df["display_label"].value_counts()
        print(f"\nDisplay labels created: {dict(label_counts)}")
    
    # Aggregate results if we have multiple runs per configuration
    print("Aggregating results...")
    df_agg = aggregate_results(df)
    print(f"Aggregated to {len(df_agg)} entries")
    
    # Get unique configs from aggregated data (filtered and sorted)
    unique_configs_raw = df_agg["display_label"].unique()
    configs = sorted([str(c) for c in unique_configs_raw if pd.notna(c) and str(c) != "nan"])
    print(f"Found {len(configs)} configurations: {', '.join(configs)}")
    
    data_sizes = sorted(df["data_size"].unique())
    print(f"Found {len(data_sizes)} data sizes: {data_sizes}")
    
    # Show sample counts per configuration
    counts = df.groupby(["display_label", "data_size"]).size()
    print(f"\nRuns per (config, size): min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    # Set seaborn style to darkgrid
    sns.set_style("darkgrid")
    
    # Build metrics list - all in seconds now
    metrics = [
        ("build_time_seconds", "Tree Construction Time", "Time (seconds)", "s"),
    ]
    
    # Add proof time if available
    if "proof_time_seconds_mean" in df_agg.columns:
        if df_agg["proof_time_seconds_mean"].sum() > 0:
            metrics.append(("proof_time_seconds", "Proof Generation Time", "Time (seconds)", "s"))
    
    # Add verify time if available and meaningful
    if "verify_time_seconds_mean" in df_agg.columns:
        if df_agg["verify_time_seconds_mean"].sum() > 0:
            metrics.append(("verify_time_seconds", "Verification Time", "Time (seconds)", "s"))
    
    # Create figure with appropriate number of subplots
    num_plots = len(metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
    
    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    
    # Color palette - assign colors dynamically based on unique configs
    palette = sns.color_palette("husl", n_colors=len(configs))
    config_colors = {config: palette[i] for i, config in enumerate(configs)}
    
    # Marker styles - use different markers for different implementations
    config_markers = {}
    marker_list = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P"]
    impl_markers = {}
    for config in configs:
        # Extract implementation if available
        config_df = df_agg[df_agg["display_label"] == config]
        if not config_df.empty and "implementation" in df_agg.columns:
            impl = config_df["implementation"].iloc[0]
            if pd.notna(impl):
                impl = str(impl)
            else:
                impl = "unknown"
        else:
            impl = "unknown"
        if impl not in impl_markers:
            impl_markers[impl] = marker_list[len(impl_markers) % len(marker_list)]
        config_markers[config] = impl_markers[impl]
    
    for idx, (metric_col, title, ylabel, unit) in enumerate(metrics):
        ax = axes[idx]
        
        # Plot each configuration
        for config_name in sorted(configs):
            config_data = df_agg[df_agg["display_label"] == config_name].sort_values("data_size")
            
            if config_data.empty:
                continue
            
            color = config_colors.get(config_name, sns.color_palette("tab10")[idx])
            marker = config_markers.get(config_name, "o")
            
            line_label = config_name if idx == 0 else None
            
            mean_col = f"{metric_col}_mean"
            std_col = f"{metric_col}_std"
            
            # Plot mean line
            ax.plot(
                config_data["data_size"],
                config_data[mean_col],
                marker=marker,
                linewidth=2.5,
                markersize=8,
                label=line_label,
                color=color,
                alpha=0.9,
            )
            
            # Add shaded area for standard deviation
            if std_col in config_data.columns and not config_data[std_col].isna().all():
                ax.fill_between(
                    config_data["data_size"],
                    config_data[mean_col] - config_data[std_col],
                    config_data[mean_col] + config_data[std_col],
                    alpha=0.2,
                    color=color,
                    label=None,
                )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Data Size (log scale)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{ylabel} (log scale)", fontsize=12, fontweight="bold")
        
        # Use log scale for both axes to match multiplicative x-axis
        ax.set_xscale("log", base=args.base_size)
        
        # Always use log scale for y-axis to properly visualize multiplicative relationships
        mean_col = f"{metric_col}_mean"
        y_values = df_agg[mean_col].values
        
        # Filter out zero and negative values for log scale
        positive_values = y_values[y_values > 0]
        if len(positive_values) == 0:
            print(f"Warning: No positive values for {metric_col}, skipping log scale")
            ax.set_yscale("linear")
        else:
            y_min = positive_values.min()
            y_max = positive_values.max()
            
            # Use log scale for y-axis
            ax.set_yscale("log")
            
            # Set reasonable y-axis limits with some padding in log space
            # Padding factor: expand range by ~20% in log space
            log_min = np.log10(y_min)
            log_max = np.log10(y_max)
            log_range = log_max - log_min
            padding = log_range * 0.1  # 10% padding
            
            ax.set_ylim(10**(log_min - padding), 10**(log_max + padding))
        
        # Enhanced grid: major and minor grid lines for better readability
        # Major grid lines (more prominent) - visible on darkgrid background
        ax.grid(True, which="major", alpha=0.7, linestyle="-", linewidth=1.8, color="gray", zorder=0)
        # Minor grid lines (subtle but visible) - help see multiplicative spacing
        ax.grid(True, which="minor", alpha=0.4, linestyle=":", linewidth=1.0, color="lightgray", zorder=0)
        
        # Ensure plot elements are above grid
        for line in ax.lines:
            line.set_zorder(3)
        for collection in ax.collections:
            collection.set_zorder(2)
        
        # Format x-axis labels
        unique_sizes = sorted(df_agg["data_size"].unique())
        if len(unique_sizes) <= 10:
            tick_sizes = unique_sizes
        else:
            # Show every nth tick for readability
            step = max(1, len(unique_sizes) // 10)
            tick_sizes = unique_sizes[::step]
        
        ax.set_xticks(tick_sizes)
        labels = [format_size_label(s, args.base_size) for s in tick_sizes]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        
        # Format y-axis to show clear log scale with appropriate tick locations
        # Use matplotlib's automatic log scale formatting but ensure visibility
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.2)
        ax.tick_params(axis="both", which="minor", labelsize=8, width=0.8)
        
        # Add a subtle note about log scale in the title or as text
        # The title already indicates it, but we can make axes more explicit
        
        # Only show legend on the first subplot
        if idx == 0:
            ax.legend(title="Configuration", loc="best", framealpha=0.9)
    
    # Add overall title with log scale indication
    fig.suptitle(
        "Dimension Comparison: Tree Performance Across Different Configurations\n"
        "(Both axes use logarithmic scale)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    
    # Save plot
    print(f"Saving plot to {args.output}...")
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"✓ Plot saved successfully!")
    
    # Print summary statistics
    print("\n=== Summary Statistics (Mean ± Std) ===")
    for config_name in sorted(configs):
        config_data = df_agg[df_agg["display_label"] == config_name]
        if config_data.empty:
            continue
        
        print(f"\n{config_name}:")
        if "build_time_seconds_mean" in config_data.columns:
            print(f"  Build time: {config_data['build_time_seconds_mean'].min():.6f} - {config_data['build_time_seconds_mean'].max():.6f} s")
        if "proof_time_seconds_mean" in config_data.columns:
            print(f"  Proof time: {config_data['proof_time_seconds_mean'].min():.6f} - {config_data['proof_time_seconds_mean'].max():.6f} s")
        if "verify_time_seconds_mean" in config_data.columns:
            print(f"  Verify time: {config_data['verify_time_seconds_mean'].min():.6f} - {config_data['verify_time_seconds_mean'].max():.6f} s")
        
        # Show which config is fastest at each size
        for size in sorted(data_sizes):
            size_data = df_agg[df_agg["data_size"] == size]
            if not size_data.empty and "build_time_seconds_mean" in size_data.columns:
                fastest_build = size_data.loc[size_data["build_time_seconds_mean"].idxmin()]
                if fastest_build["display_label"] == config_name:
                    mean = fastest_build["build_time_seconds_mean"]
                    std = fastest_build.get("build_time_seconds_std", 0)
                    print(f"  ⚡ Fastest build at {format_size_label(size, args.base_size)}: {mean:.6f} ± {std:.6f} s")


if __name__ == "__main__":
    main()

