#!/usr/bin/env python3
"""
Generate budget cases for benchmarking based on actual model costs.

Budgets are generated to be around 1/4 of total model cost, allowing
approximately 1/4 of layers to be covered.
"""

import json
import numpy as np
from pathlib import Path
import argparse


def get_model_costs(scores_dir="../scores"):
    """
    Calculate total costs for all models to determine realistic budget ranges.
    
    Returns:
        Dictionary mapping model names to (total_cost, avg_cost_per_layer)
    """
    scores_path = Path(scores_dir)
    model_costs = {}
    
    # Model score files
    model_files = [
        ("llama2-70b_scores.json", "llama2_70b"),
        ("llama2-13b_scores.json", "llama2_13b"),
        ("llama2-7b_scores.json", "llama2_7b"),
        ("opt125m_scores.json", "opt_125m"),
    ]
    
    for score_file, model_name in model_files:
        file_path = scores_path / score_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            costs = [layer['num_parameters'] for layer in data['layers']]
            total_cost = sum(costs)
            avg_cost = total_cost / len(costs) if costs else 0
            model_costs[model_name] = {
                'total_cost': total_cost,
                'avg_cost_per_layer': avg_cost,
                'num_layers': len(costs)
            }
    
    return model_costs


def generate_budget_cases(num_cases=100, num_verifiers=3, output_dir="cases", scores_dir="../scores"):
    """
    Generate diverse budget cases based on actual model costs.
    
    Budgets are set to be around 1/4 of total model cost (allowing ~1/4 of layers
    to be covered), with variation to test different scenarios.
    
    Args:
        num_cases: Number of budget cases to generate
        num_verifiers: Number of verifiers
        output_dir: Directory to save budget cases
        scores_dir: Directory containing score files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model costs to determine realistic budget ranges
    model_costs = get_model_costs(scores_dir)
    
    if not model_costs:
        print("Warning: No model cost data found. Using default ranges.")
        # Fallback: use average of typical model costs
        # Approximate: average of the 4 models
        avg_total_cost = (64_760_053_760 + 12_687_769_600 + 6_476_005_376 + 84_934_656) / 4
        target_quarter = avg_total_cost / 4
    else:
        # Use the average of all models' quarter costs as baseline
        quarter_costs = [info['total_cost'] / 4 for info in model_costs.values()]
        target_quarter = np.mean(quarter_costs)
        print(f"Model cost analysis:")
        for model, info in model_costs.items():
            print(f"  {model}: total={info['total_cost']:,.0f}, "
                  f"1/4={info['total_cost']/4:,.0f}, "
                  f"avg/layer={info['avg_cost_per_layer']:,.0f}")
        print(f"\nTarget budget range (around 1/4 of total cost): {target_quarter:,.0f}")
    
    np.random.seed(42)  # For reproducibility
    
    cases = []
    
    for case_id in range(1, num_cases + 1):
        # Generate budgets around 1/4 of total cost with variation
        # Range: 0.15 to 0.35 of total cost (centered around 0.25)
        budget_ratio = np.random.uniform(0.15, 0.35)
        total_budget = target_quarter * (budget_ratio / 0.25)  # Scale around target
        
        # Generate diverse budget scenarios
        # Strategy 1: Uniform random budgets (20% of cases)
        if case_id <= num_cases * 0.2:
            budgets = np.random.uniform(0.1 * total_budget / num_verifiers, 
                                       2.0 * total_budget / num_verifiers, 
                                       num_verifiers)
            budgets = budgets / budgets.sum() * total_budget
        
        # Strategy 2: One dominant verifier (20% of cases)
        elif case_id <= num_cases * 0.4:
            budgets = np.ones(num_verifiers) * total_budget / num_verifiers
            dominant_idx = np.random.randint(0, num_verifiers)
            budgets[dominant_idx] = total_budget * 0.7
            budgets = budgets / budgets.sum() * total_budget
        
        # Strategy 3: Decreasing budgets (20% of cases)
        elif case_id <= num_cases * 0.6:
            ratios = np.exp(-np.linspace(0, 2, num_verifiers))
            budgets = ratios / ratios.sum() * total_budget
        
        # Strategy 4: Increasing budgets (20% of cases)
        elif case_id <= num_cases * 0.8:
            ratios = np.exp(np.linspace(0, 2, num_verifiers))
            budgets = ratios / ratios.sum() * total_budget
        
        # Strategy 5: Equal budgets (20% of cases)
        else:
            budgets = np.ones(num_verifiers) * total_budget / num_verifiers
        
        # Round to reasonable precision
        budgets = np.round(budgets, 2)
        
        case_data = {
            'case_id': case_id,
            'num_verifiers': num_verifiers,
            'budgets': budgets.tolist(),
            'total_budget': float(budgets.sum())
        }
        
        cases.append(case_data)
        
        # Save individual case file
        case_file = output_path / f"case_{case_id:03d}.json"
        with open(case_file, 'w') as f:
            json.dump(case_data, f, indent=2)
    
    # Save summary file with all cases
    summary_file = output_path / "all_cases.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'num_cases': num_cases,
            'num_verifiers': num_verifiers,
            'cases': cases
        }, f, indent=2)
    
    print(f"Generated {num_cases} budget cases in {output_path}")
    print(f"Summary saved to {summary_file}")
    
    return cases


def main():
    parser = argparse.ArgumentParser(description='Generate budget cases for benchmarking')
    parser.add_argument('--num_cases', type=int, default=100,
                       help='Number of budget cases to generate')
    parser.add_argument('--num_verifiers', type=int, default=3,
                       help='Number of verifiers')
    parser.add_argument('--output_dir', type=str, default='cases',
                       help='Output directory for budget cases')
    parser.add_argument('--scores_dir', type=str, default='../layerSelectionLib/scores',
                       help='Directory containing score files for cost analysis')
    
    args = parser.parse_args()
    
    generate_budget_cases(
        num_cases=args.num_cases,
        num_verifiers=args.num_verifiers,
        output_dir=args.output_dir,
        scores_dir=args.scores_dir
    )


if __name__ == '__main__':
    main()

