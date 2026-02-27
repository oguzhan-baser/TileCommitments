#!/usr/bin/env python3
"""
Run optimization benchmarks with different objective functions.

Compares 5 different optimization strategies:
1. Random (random integer values for each layer)
2. WeightNorm 1 (L1 norm as value)
3. WeightNorm 2 (L2 norm as value)
4. WeightNorm Inf (L∞ norm as value)
5. Classic alphaprun (significance_score as value)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add layerSelectionLib so OptimizationModule can be imported
experiments_dir = Path(__file__).parent
layer_selection_lib = experiments_dir.parent / "layerSelectionLib"
if str(layer_selection_lib) not in sys.path:
    sys.path.insert(0, str(layer_selection_lib))

from OptimizationModule.layer_selection_optimizer import LayerSelectionOptimizer
from OptimizationModule.scores_loader import load_scores_from_json


def load_norms(norms_file):
    """Load weight norms from JSON file."""
    with open(norms_file, 'r') as f:
        norms_data = json.load(f)
    
    # Create mapping from layer index to norms
    norms_dict = {}
    for layer in norms_data['layers']:
        idx = layer['index']
        norms_dict[idx] = {
            'norm_1': layer['norm_1'],
            'norm_2': layer['norm_2'],
            'norm_inf': layer['norm_inf']
        }
    
    return norms_dict, norms_data


def get_objective_values(method, scores_data, norms_dict=None, random_seed=None):
    """
    Get objective values for a given method.
    
    Args:
        method: One of 'random', 'norm_1', 'norm_2', 'norm_inf', 'alphaprun'
        scores_data: Scores JSON data
        norms_dict: Dictionary mapping layer index to norms (optional)
        random_seed: Random seed for random method (optional)
    
    Returns:
        List of objective values for each layer
    """
    layers = scores_data['layers']
    values = []
    
    # Set random seed if provided (for reproducibility of random method)
    if method == 'random' and random_seed is not None:
        np.random.seed(random_seed)
    
    num_layers = len(layers)
    
    for i, layer in enumerate(layers):
        idx = layer['index']
        
        if method == 'random':
            # Assign random integer value to each layer
            # But weight it so middle layers get higher values and edge layers get lower values
            # Use a bell curve-like distribution: higher in middle, lower at edges
            # Normalize position to [0, 1] where 0 is first layer and 1 is last layer
            position = i / (num_layers - 1) if num_layers > 1 else 0.5
            
            # Create a weight function that peaks in the middle
            # Using a quadratic function: 1 - 4*(x - 0.5)^2, which gives 1 at middle and 0 at edges
            # Or use a smoother curve: 1 - (2*x - 1)^2
            weight = 1.0 - (2.0 * position - 1.0) ** 2  # Peaks at 1.0 in middle, 0.0 at edges
            
            # Scale weight to be between 0.1 and 1.0 (so edges still get some value)
            weight = 0.1 + 0.9 * weight
            
            # Generate random integer and scale by weight
            base_random = np.random.randint(1, 1000000)
            weighted_value = int(base_random * weight)
            
            values.append(float(max(1, weighted_value)))  # Ensure at least 1
        
        elif method == 'norm_1':
            if norms_dict and idx in norms_dict:
                values.append(1/norms_dict[idx]['norm_1']) # TODO changed
            else:
                values.append(0.0)
        
        elif method == 'norm_2':
            if norms_dict and idx in norms_dict:
                values.append(1/norms_dict[idx]['norm_2']) # TODO changed
            else:
                values.append(0.0)
        
        elif method == 'norm_inf':
            if norms_dict and idx in norms_dict:
                values.append(norms_dict[idx]['norm_inf'])
            else:
                values.append(0.0)
        
        elif method == 'alphaprun':
            values.append(float(layer['significance_score']))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return values


def get_total_alphaprun_score(scores_data):
    """
    Calculate total alphaprun score (sum of all layers' significance_score).
    
    Args:
        scores_data: Scores JSON data with significance_score for each layer
    
    Returns:
        Total alphaprun score across all layers
    """
    total_score = 0.0
    for layer in scores_data['layers']:
        total_score += layer['significance_score']
    return total_score


def evaluate_with_alphaprun_objective(assignments, scores_data):
    """
    Evaluate assignments using the original alphaprun objective (significance_score).
    
    Args:
        assignments: List of (start, end) tuples for each verifier
        scores_data: Scores JSON data with significance_score for each layer
    
    Returns:
        Total alphaprun benefit
    """
    total_benefit = 0.0
    
    for start, end in assignments:
        if start == -1 or end == -1:
            continue
        
        for idx in range(start, end + 1):
            if idx < len(scores_data['layers']):
                layer = scores_data['layers'][idx]
                total_benefit += layer['significance_score']
    
    return total_benefit


def run_benchmark_case(scores_file, norms_file, budgets, methods=None, case_id=None, total_alphaprun_score=None):
    """
    Run optimization for a single budget case with all methods.
    
    Args:
        scores_file: Path to scores JSON file
        norms_file: Path to norms JSON file (or None if not available)
        budgets: List of verifier budgets
        methods: List of methods to run (default: all 5)
        case_id: Case ID for random seed (optional)
        total_alphaprun_score: Total alphaprun score for the model (optional, calculated if not provided)
    
    Returns:
        Dictionary with results for each method
    """
    if methods is None:
        methods = ['random', 'norm_1', 'norm_2', 'norm_inf', 'alphaprun']
    
    # Load scores
    with open(scores_file, 'r') as f:
        scores_data = json.load(f)
    
    # Calculate total alphaprun score if not provided
    if total_alphaprun_score is None:
        total_alphaprun_score = get_total_alphaprun_score(scores_data)
    
    # Load norms if available
    norms_dict = None
    if norms_file and Path(norms_file).exists():
        norms_dict, _ = load_norms(norms_file)
    
    # Get costs (num_parameters)
    costs = [layer['num_parameters'] for layer in scores_data['layers']]
    
    results = {}
    
    for method in methods:
        # Skip norm methods if norms not available
        if method in ['norm_1', 'norm_2', 'norm_inf'] and norms_dict is None:
            print(f"Warning: Skipping {method} - norms file not available")
            continue
        
        # Get objective values for this method
        # Use case_id as random seed for random method to ensure reproducibility
        random_seed = case_id if case_id is not None else None
        values = get_objective_values(method, scores_data, norms_dict, random_seed=random_seed)
        
        # Run optimization
        try:
            optimizer = LayerSelectionOptimizer(values, costs, budgets)
            optimal_benefit, assignments = optimizer.solve()
            
            # Verify solution
            is_valid, error_msg = optimizer.verify_solution(assignments)
            if not is_valid:
                print(f"Warning: Invalid solution for {method}: {error_msg}")
            
            # Evaluate with alphaprun objective
            alphaprun_benefit = evaluate_with_alphaprun_objective(assignments, scores_data)
            
            # Calculate ratio: achieved benefit / total alphaprun score
            alphaprun_ratio = alphaprun_benefit / total_alphaprun_score if total_alphaprun_score > 0 else 0.0
            
            # Convert assignments to JSON-serializable format (numpy int64 -> int)
            assignments_serializable = [
                (int(start), int(end)) for start, end in assignments
            ]
            
            results[method] = {
                'optimal_benefit': float(optimal_benefit),
                'alphaprun_benefit': float(alphaprun_benefit),  # Keep raw benefit for reference
                'alphaprun_ratio': float(alphaprun_ratio),  # New: ratio as default
                'assignments': assignments_serializable,
                'is_valid': is_valid
            }
        
        except Exception as e:
            print(f"Error running {method}: {e}")
            results[method] = {
                'error': str(e)
            }
    
    return results


def run_all_benchmarks(scores_file, norms_file, cases_dir, output_dir, model_name=None):
    """
    Run benchmarks for all cases.
    
    Budgets are scaled proportionally to the model's cost to ensure they're
    appropriate for the model size (around 1/4 of total cost).
    
    Args:
        scores_file: Path to scores JSON file
        norms_file: Path to norms JSON file
        cases_dir: Directory containing budget cases
        output_dir: Directory to save results
        model_name: Model name (for output file naming)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load scores to get model cost
    with open(scores_file, 'r') as f:
        scores_data = json.load(f)
    
    # Calculate model total cost
    model_costs = [layer['num_parameters'] for layer in scores_data['layers']]
    model_total_cost = sum(model_costs)
    model_quarter_cost = model_total_cost / 4
    
    # Calculate total alphaprun score for the model (used for ratio calculation)
    total_alphaprun_score = get_total_alphaprun_score(scores_data)
    
    # Load all cases
    cases_file = Path(cases_dir) / "all_cases.json"
    if not cases_file.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_file}")
    
    with open(cases_file, 'r') as f:
        cases_data = json.load(f)
    
    # Calculate scaling factor based on model cost
    # Use random budget ratios between 0.2 and 0.4 for each case (high variance)
    # Each case will have a different budget ratio sampled uniformly from [0.2, 0.4]
    min_budget_ratio = 0.2
    max_budget_ratio = 0.4
    
    # Get reference total budget from first case
    reference_total_budget = sum(cases_data['cases'][0]['budgets'])
    
    # Set random seed for reproducibility (based on model name hash)
    # if model_name:
    #     seed_value = hash(model_name) % (2**32)
    # else:
    seed_value = 42
    np.random.seed(seed_value)
    
    print(f"Model cost scaling (using random ratios 0.2-0.4):")
    print(f"  Model total cost: {model_total_cost:,.0f}")
    print(f"  Model 1/4 cost: {model_quarter_cost:,.0f}")
    print(f"  Budget ratio range: [{min_budget_ratio}, {max_budget_ratio}]")
    print(f"  Each case will use a random ratio in this range")
    
    all_results = []
    
    print(f"\nRunning benchmarks for {len(cases_data['cases'])} cases...")
    
    for case in cases_data['cases']:
        case_id = case['case_id']
        original_budgets = case['budgets']
        
        # Generate random budget ratio for this case (between 0.2 and 0.4)
        # Use case_id as part of seed for reproducibility while maintaining variance
        np.random.seed(seed_value + case_id)
        case_budget_ratio = np.random.uniform(min_budget_ratio, max_budget_ratio)
        
        # Target: scaled budgets should sum to case_budget_ratio * model_quarter_cost
        target_total_budget = case_budget_ratio * model_quarter_cost
        
        # Scale factor: target total budget / reference total budget
        scale_factor = target_total_budget / reference_total_budget
        
        # Scale budgets to match model size with this case's ratio
        scaled_budgets = [b * scale_factor for b in original_budgets]
        
        print(f"Running case {case_id}/{len(cases_data['cases'])}... (ratio: {case_budget_ratio:.3f})")
        
        results = run_benchmark_case(scores_file, norms_file, scaled_budgets, 
                                    case_id=case_id, total_alphaprun_score=total_alphaprun_score)
        
        # Convert scaled budgets to native Python types (in case they're numpy)
        budgets_serializable = [float(b) for b in scaled_budgets]
        
        result_entry = {
            'case_id': case_id,
            'budgets': budgets_serializable,
            'original_budgets': [float(b) for b in original_budgets],  # Keep original for reference
            'budget_ratio': float(case_budget_ratio),  # Store the ratio used for this case
            'scale_factor': float(scale_factor),
            'results': results
        }
        
        all_results.append(result_entry)
        
        # Save individual case result
        case_output = output_path / f"case_{case_id:03d}_results.json"
        with open(case_output, 'w') as f:
            json.dump(result_entry, f, indent=2)
    
    # Save summary
    summary_output = output_path / "all_results.json"
    if model_name:
        summary_output = output_path / f"{model_name}_all_results.json"
    
    with open(summary_output, 'w') as f:
        json.dump({
            'model_name': model_name,
            'scores_file': str(scores_file),
            'norms_file': str(norms_file),
            'num_cases': len(all_results),
            'total_alphaprun_score': float(total_alphaprun_score),  # Store total score for reference
            'results': all_results
        }, f, indent=2)
    
    print(f"\nBenchmark complete! Results saved to {output_path}")
    print(f"Summary saved to {summary_output}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run optimization benchmarks')
    parser.add_argument('--scores', type=str, required=True,
                       help='Path to scores JSON file')
    parser.add_argument('--norms', type=str, default=None,
                       help='Path to norms JSON file')
    parser.add_argument('--cases_dir', type=str, default='cases',
                       help='Directory containing budget cases')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (for output file naming)')
    
    args = parser.parse_args()
    
    # Extract model name from scores file if not provided
    if args.model_name is None:
        with open(args.scores, 'r') as f:
            scores_data = json.load(f)
        args.model_name = scores_data.get('model_name', 'unknown')
    
    run_all_benchmarks(
        args.scores,
        args.norms,
        args.cases_dir,
        args.output_dir,
        args.model_name
    )


if __name__ == '__main__':
    main()

