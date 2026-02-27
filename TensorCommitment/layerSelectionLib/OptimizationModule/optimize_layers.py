#!/usr/bin/env python3
"""
Main script to run layer selection optimization on scores files.

Usage:
    python optimize_layers.py --scores scores/opt125m_scores.json --budgets 10000000 20000000 15000000
"""

import argparse
import json
import sys
from pathlib import Path

import sys
from pathlib import Path

# Add OptimizationModule to path
module_path = Path(__file__).parent
if str(module_path) not in sys.path:
    sys.path.insert(0, str(module_path.parent))

from OptimizationModule.layer_selection_optimizer import LayerSelectionOptimizer
from OptimizationModule.scores_loader import load_scores_from_json


def format_assignment(start: int, end: int, layer_names: list = None) -> str:
    """Format an assignment for display."""
    if start == -1 and end == -1:
        return "Empty (no layers assigned)"
    if layer_names:
        return f"[{start}, {end}] ({layer_names[start]} ... {layer_names[end]})"
    return f"[{start}, {end}]"


def main():
    parser = argparse.ArgumentParser(
        description='Optimize layer selection for multi-verifier budgeted contiguous selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--scores', '--scores_file', type=str, required=True,
                       dest='scores',
                       help='Path to JSON file with layer scores (values and costs)')
    parser.add_argument('--budgets', type=float, nargs='+', required=True,
                       help='Budget for each verifier (space-separated list)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    # Load scores
    print(f"Loading scores from {args.scores}...")
    try:
        values, costs, metadata = load_scores_from_json(args.scores)
        print(f"Loaded {len(values)} layers from model {metadata['model_name']}")
    except Exception as e:
        print(f"Error loading scores: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load layer names for better output
    with open(args.scores, 'r') as f:
        scores_data = json.load(f)
    layer_names = [layer['name'] for layer in scores_data.get('layers', [])]
    
    # Validate budgets
    if len(args.budgets) == 0:
        print("Error: At least one budget must be provided", file=sys.stderr)
        sys.exit(1)
    
    budgets = args.budgets
    print(f"Using {len(budgets)} verifiers with budgets: {budgets}")
    
    # Run optimization
    print("\nRunning optimization...")
    try:
        optimizer = LayerSelectionOptimizer(values, costs, budgets)
        optimal_benefit, assignments = optimizer.solve()
        
        # Verify solution
        is_valid, error_msg = optimizer.verify_solution(assignments)
        if not is_valid:
            print(f"Warning: Solution verification failed: {error_msg}", file=sys.stderr)
        
        computed_benefit = optimizer.compute_total_benefit(assignments)
        
    except Exception as e:
        print(f"Error during optimization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Optimal total benefit: {optimal_benefit:.6f}")
    print(f"Computed benefit: {computed_benefit:.6f}")
    print(f"Solution valid: {is_valid}")
    print("\nLayer assignments:")
    
    total_cost = 0.0
    total_benefit = 0.0
    
    for k, (start, end) in enumerate(assignments):
        if start == -1 and end == -1:
            print(f"  Verifier {k+1}: Empty (budget: {budgets[k]:.0f})")
        else:
            interval_cost = optimizer.C[end + 1] - optimizer.C[start]
            interval_benefit = optimizer.V[end + 1] - optimizer.V[start]
            total_cost += interval_cost
            total_benefit += interval_benefit
            
            print(f"  Verifier {k+1}: Layers {start}-{end} "
                  f"(cost: {interval_cost:.0f}/{budgets[k]:.0f}, "
                  f"benefit: {interval_benefit:.6f})")
            if layer_names:
                print(f"    {layer_names[start]} ... {layer_names[end]}")
    
    print(f"\nTotal cost: {total_cost:.0f}")
    print(f"Total benefit: {total_benefit:.6f}")
    print("="*70)
    
    # Prepare output data
    output_data = {
        'input': {
            'scores_file': args.scores,
            'model_name': metadata['model_name'],
            'num_layers': len(values),
            'budgets': budgets,
            'num_verifiers': len(budgets)
        },
        'results': {
            'optimal_benefit': optimal_benefit,
            'computed_benefit': computed_benefit,
            'is_valid': is_valid,
            'total_cost': total_cost,
            'total_benefit': total_benefit,
            'assignments': []
        }
    }
    
    for k, (start, end) in enumerate(assignments):
        assignment_info = {
            'verifier': int(k + 1),
            'start_layer': int(start),
            'end_layer': int(end),
            'budget': budgets[k],
            'cost': 0.0,
            'benefit': 0.0,
            'layers': []
        }
        
        if start != -1 and end != -1:
            assignment_info['cost'] = float(optimizer.C[end + 1] - optimizer.C[start])
            assignment_info['benefit'] = float(optimizer.V[end + 1] - optimizer.V[start])
            assignment_info['layers'] = [
                {
                    'index': int(idx),
                    'name': layer_names[idx],
                    'value': float(values[idx]),
                    'cost': float(costs[idx])
                }
                for idx in range(start, end + 1)
            ]
        
        output_data['results']['assignments'].append(assignment_info)
    
    # Save output if requested
    if args.output:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, 'w') as f:
            if args.pretty:
                json.dump(output_data, f, indent=2)
            else:
                json.dump(output_data, f)
        print("Saved!")
    
    return output_data


if __name__ == '__main__':
    main()
