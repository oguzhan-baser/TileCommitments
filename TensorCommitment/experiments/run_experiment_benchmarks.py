#!/usr/bin/env python3
"""
Run benchmarks on experiment cases and generate plots.

Takes experiment cases and runs optimization benchmarks for all models,
then generates plots comparing results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

# Add benchmark directory to path for imports
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from run_benchmarks import run_benchmark_case, get_total_alphaprun_score


def run_experiment_benchmarks(experiment_cases_file, scores_dir, norms_dir, 
                            output_dir, models=None):
    """
    Run benchmarks on experiment cases for all models.
    
    Args:
        experiment_cases_file: Path to experiment cases JSON file
        scores_dir: Directory containing score files
        norms_dir: Directory containing norm files
        output_dir: Directory to save results
        models: List of model names to process (None = all)
    """
    with open(experiment_cases_file, 'r') as f:
        experiment_data = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Model configuration
    model_configs = {
        'llama2_70b': {
            'scores_file': 'llama2-70b_scores.json',
            'norms_file': 'llama2_70b_norms.json'
        },
        'llama2_13b': {
            'scores_file': 'llama2-13b_scores.json',
            'norms_file': 'llama2_13b_norms.json'
        },
        'llama2_7b': {
            'scores_file': 'llama2-7b_scores.json',
            'norms_file': 'llama2_7b_norms.json'
        },
        'opt_125m': {
            'scores_file': 'opt125m_scores.json',
            'norms_file': 'opt_125m_norms.json'
        }
    }
    
    if models is None:
        models = list(model_configs.keys())
    
    scores_path = Path(scores_dir)
    norms_path = Path(norms_dir)
    
    all_results = {}
    
    for model_name in models:
        if model_name not in model_configs:
            print(f"Warning: Unknown model {model_name}, skipping")
            continue
        
        config = model_configs[model_name]
        scores_file = scores_path / config['scores_file']
        norms_file = norms_path / config['norms_file']
        
        if not scores_file.exists():
            print(f"Warning: Scores file not found: {scores_file}, skipping {model_name}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")
        
        # Load scores to get total alphaprun score
        with open(scores_file, 'r') as f:
            scores_data = json.load(f)
        total_alphaprun_score = get_total_alphaprun_score(scores_data)
        
        # Calculate model costs for scaling
        model_costs = [layer['num_parameters'] for layer in scores_data['layers']]
        model_total_cost = sum(model_costs)
        model_quarter_cost = model_total_cost / 4
        
        # Get reference budget from first experiment case
        first_case = experiment_data['cases'][0]
        reference_total_budget = sum(first_case['budgets'])
        
        # Scale budgets to match model size (using 0.25 ratio as baseline)
        target_budget_ratio = 0.25
        target_total_budget = target_budget_ratio * model_quarter_cost
        scale_factor = target_total_budget / reference_total_budget
        
        print(f"Model cost: {model_total_cost:,.0f}")
        print(f"Quarter cost: {model_quarter_cost:,.0f}")
        print(f"Scale factor: {scale_factor:.4f}")
        print()
        
        case_results = []
        
        for case_idx, case in enumerate(experiment_data['cases'], 1):
            case_id = case['case_id']
            original_budgets = case['budgets']
            
            # Scale budgets to match model size
            scaled_budgets = [b * scale_factor for b in original_budgets]
            
            if case_idx % 50 == 0 or case_idx == 1:
                print(f"Running case {case_idx}/{len(experiment_data['cases'])}...")
            
            # Run benchmark
            results = run_benchmark_case(
                str(scores_file),
                str(norms_file) if norms_file.exists() else None,
                scaled_budgets,
                case_id=case_id,
                total_alphaprun_score=total_alphaprun_score
            )
            
            case_result = {
                'case_id': case_id,
                'original_case_id': case.get('original_case_id'),
                'experiment_index': case.get('experiment_index'),
                'budgets': [float(b) for b in scaled_budgets],
                'original_budgets': [float(b) for b in original_budgets],
                'scale_factor': float(scale_factor),
                'results': results
            }
            
            case_results.append(case_result)
        
        # Save results for this model
        model_output_file = output_path / f"{model_name}_experiment_results.json"
        with open(model_output_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'scores_file': str(scores_file),
                'norms_file': str(norms_file) if norms_file.exists() else None,
                'num_cases': len(case_results),
                'total_alphaprun_score': float(total_alphaprun_score),
                'source_experiment_file': str(experiment_cases_file),
                'results': case_results
            }, f, indent=2)
        
        all_results[model_name] = model_output_file
        print(f"\nSaved results to: {model_output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks on experiment cases'
    )
    parser.add_argument('--experiment_cases', type=str, required=True,
                       help='Path to experiment cases JSON file')
    parser.add_argument('--scores_dir', type=str, default='../scores',
                       help='Directory containing score files')
    parser.add_argument('--norms_dir', type=str, default='weight_norms',
                       help='Directory containing norm files')
    parser.add_argument('--output_dir', type=str, default='results/experiment',
                       help='Output directory for results')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to process (default: all)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots after running benchmarks')
    
    args = parser.parse_args()
    
    experiment_cases_file = Path(args.experiment_cases)
    if not experiment_cases_file.exists():
        print(f"Error: Experiment cases file not found: {experiment_cases_file}")
        return
    
    print("Running benchmarks on experiment cases...")
    print(f"Experiment cases file: {experiment_cases_file}")
    print()
    
    results_files = run_experiment_benchmarks(
        experiment_cases_file,
        args.scores_dir,
        args.norms_dir,
        args.output_dir,
        args.models
    )
    
    # Generate plots if requested
    if args.plot:
        print("\n" + "="*70)
        print("Generating plots...")
        print("="*70)
        
        plot_script = Path(__file__).parent / 'plot_results.py'
        output_path = Path(args.output_dir)
        
        # Generate individual plots
        for model_name, results_file in results_files.items():
            try:
                plot_output = results_file.with_suffix('.png')
                cmd = [
                    sys.executable,
                    str(plot_script),
                    '--results', str(results_file),
                    '--output', str(plot_output)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  {model_name}: Plot saved to {plot_output}")
            except Exception as e:
                print(f"  {model_name}: Warning - Could not generate plot: {e}")
        
        # Generate combined plot
        if len(results_files) > 0:
            try:
                combined_plot = output_path / 'combined_experiment_comparison.png'
                files_str = ','.join(str(f) for f in results_files.values())
                cmd = [
                    sys.executable,
                    str(plot_script),
                    '--results', files_str,
                    '--output', str(combined_plot),
                    '--combined'
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"\nCombined plot saved to: {combined_plot}")
            except subprocess.CalledProcessError as e:
                print(f"\nWarning: Could not generate combined plot:")
                if e.stderr:
                    print(f"  {e.stderr}")
            except Exception as e:
                print(f"\nWarning: Could not generate combined plot: {e}")
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

