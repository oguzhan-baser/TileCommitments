#!/usr/bin/env python3
"""
Generate scalability test cases with varying numbers of verifiers.

Each verifier gets 1% of total model parameters.
Tests verifier counts: 1, 6, 11, 16, 21, ..., 96 (increment by 5).
"""

import argparse
import json
import numpy as np
from pathlib import Path


def get_model_total_cost(scores_file):
    """Get total cost (parameters) for a model from scores file."""
    with open(scores_file, 'r') as f:
        scores_data = json.load(f)
    costs = [layer['num_parameters'] for layer in scores_data['layers']]
    return sum(costs)


def generate_scalability_cases(scores_dir, output_file, 
                               verifier_counts=None, 
                               budget_percentage=0.01,
                               num_cases_per_count=1):
    """
    Generate scalability test cases.
    
    Args:
        scores_dir: Directory containing score files
        output_file: Path to save cases JSON file
        verifier_counts: List of verifier counts to test (default: 1, 6, 11, ..., 96)
        budget_percentage: Percentage of total parameters per verifier (default: 0.01 = 1%)
        num_cases_per_count: Number of cases per verifier count (default: 1)
    
    Returns:
        Dictionary with cases data
    """
    scores_path = Path(scores_dir)
    
    # Default verifier counts: 1, 6, 11, 16, ..., 96
    if verifier_counts is None:
        verifier_counts = list(range(1, 101, 5))  # 1, 6, 11, ..., 96
    
    # Model score files
    model_files = {
        'llama2_70b': 'llama2-70b_scores.json',
        'llama2_13b': 'llama2-13b_scores.json',
        'llama2_7b': 'llama2-7b_scores.json',
        'opt_125m': 'opt125m_scores.json',
    }
    
    # Get total costs for each model
    model_costs = {}
    for model_name, score_file in model_files.items():
        score_path = scores_path / score_file
        if score_path.exists():
            model_costs[model_name] = get_model_total_cost(score_path)
        else:
            print(f"Warning: {score_path} not found, skipping {model_name}")
    
    if not model_costs:
        raise ValueError("No model cost data found!")
    
    # Generate cases for each model and verifier count
    all_cases = []
    case_id = 1
    
    for model_name, total_cost in model_costs.items():
        for num_verifiers in verifier_counts:
            # Each verifier gets budget_percentage of total cost
            budget_per_verifier = budget_percentage * total_cost
            
            # Create num_cases_per_count cases with the same budget
            for case_idx in range(num_cases_per_count):
                # All verifiers get the same budget
                budgets = [budget_per_verifier] * num_verifiers
                
                case = {
                    'case_id': case_id,
                    'model_name': model_name,
                    'num_verifiers': num_verifiers,
                    'budget_percentage': budget_percentage,
                    'budget_per_verifier': float(budget_per_verifier),
                    'total_budget': float(sum(budgets)),
                    'total_model_cost': float(total_cost),
                    'budgets': [float(b) for b in budgets]
                }
                
                all_cases.append(case)
                case_id += 1
    
    # Create output data
    output_data = {
        'description': 'Scalability test cases with varying numbers of verifiers',
        'budget_percentage': budget_percentage,
        'verifier_counts': verifier_counts,
        'num_cases_per_count': num_cases_per_count,
        'models': list(model_costs.keys()),
        'model_costs': {k: float(v) for k, v in model_costs.items()},
        'total_cases': len(all_cases),
        'cases': all_cases
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated {len(all_cases)} scalability cases")
    print(f"  Models: {len(model_costs)}")
    print(f"  Verifier counts: {len(verifier_counts)} ({verifier_counts[0]}, {verifier_counts[1]}, ..., {verifier_counts[-1]})")
    print(f"  Cases per count: {num_cases_per_count}")
    print(f"  Budget per verifier: {budget_percentage:.1%} of total parameters")
    print(f"  Saved to: {output_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Generate scalability test cases'
    )
    parser.add_argument('--scores_dir', type=str, default='../../scores',
                       help='Directory containing score files')
    parser.add_argument('--output', type=str, default='scalability_cases.json',
                       help='Output path for cases JSON file')
    parser.add_argument('--budget_percentage', type=float, default=0.01,
                       help='Percentage of total parameters per verifier (default: 0.01 = 1%%)')
    parser.add_argument('--num_cases_per_count', type=int, default=1,
                       help='Number of cases per verifier count (default: 1)')
    parser.add_argument('--max_verifiers', type=int, default=96,
                       help='Maximum number of verifiers (default: 96)')
    parser.add_argument('--start_verifiers', type=int, default=1,
                       help='Starting number of verifiers (default: 1)')
    parser.add_argument('--increment', type=int, default=5,
                       help='Increment between verifier counts (default: 5)')
    
    args = parser.parse_args()
    
    # Generate verifier counts
    verifier_counts = list(range(args.start_verifiers, args.max_verifiers + 1, args.increment))
    
    generate_scalability_cases(
        args.scores_dir,
        args.output,
        verifier_counts=verifier_counts,
        budget_percentage=args.budget_percentage,
        num_cases_per_count=args.num_cases_per_count
    )


if __name__ == '__main__':
    main()

