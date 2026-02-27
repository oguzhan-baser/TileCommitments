#!/usr/bin/env python3
"""
Run scalability benchmarks on generated cases.

Tests how AlphaPrun benefit ratio changes with increasing number of verifiers.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add benchmark directory to path for imports
benchmark_dir = Path(__file__).parent.parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from run_benchmarks import run_benchmark_case, get_total_alphaprun_score


def run_scalability_benchmarks(cases_file, scores_dir, norms_dir, output_dir):
    """
    Run benchmarks on scalability cases.
    
    Args:
        cases_file: Path to scalability cases JSON file
        scores_dir: Directory containing score files
        norms_dir: Directory containing norm files
        output_dir: Directory to save results
    """
    with open(cases_file, 'r') as f:
        cases_data = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scores_path = Path(scores_dir)
    norms_path = Path(norms_dir)
    
    # Model score file mapping
    model_score_files = {
        'llama2_70b': 'llama2-70b_scores.json',
        'llama2_13b': 'llama2-13b_scores.json',
        'llama2_7b': 'llama2-7b_scores.json',
        'opt_125m': 'opt125m_scores.json',
    }
    
    model_norm_files = {
        'llama2_70b': 'llama2_70b_norms.json',
        'llama2_13b': 'llama2_13b_norms.json',
        'llama2_7b': 'llama2_7b_norms.json',
        'opt_125m': 'opt_125m_norms.json',
    }
    
    # Group cases by model
    cases_by_model = {}
    for case in cases_data['cases']:
        model_name = case['model_name']
        if model_name not in cases_by_model:
            cases_by_model[model_name] = []
        cases_by_model[model_name].append(case)
    
    all_results = {}
    
    for model_name, model_cases in cases_by_model.items():
        if model_name not in model_score_files:
            print(f"Warning: Unknown model {model_name}, skipping")
            continue
        
        scores_file = scores_path / model_score_files[model_name]
        norms_file = norms_path / model_norm_files.get(model_name, '')
        
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
        
        # Sort cases by number of verifiers
        model_cases.sort(key=lambda x: x['num_verifiers'])
        
        case_results = []
        
        for case_idx, case in enumerate(model_cases, 1):
            num_verifiers = case['num_verifiers']
            budgets = case['budgets']
            
            if case_idx % 5 == 0 or case_idx == 1:
                print(f"Running case {case_idx}/{len(model_cases)} (verifiers: {num_verifiers})...")
            
            # Run benchmark
            results = run_benchmark_case(
                str(scores_file),
                str(norms_file) if norms_file.exists() else None,
                budgets,
                case_id=case['case_id'],
                total_alphaprun_score=total_alphaprun_score
            )
            
            case_result = {
                'case_id': case['case_id'],
                'num_verifiers': num_verifiers,
                'budgets': [float(b) for b in budgets],
                'total_budget': float(sum(budgets)),
                'results': results
            }
            
            case_results.append(case_result)
        
        # Save results for this model
        model_output_file = output_path / f"{model_name}_scalability_results.json"
        with open(model_output_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'scores_file': str(scores_file),
                'norms_file': str(norms_file) if norms_file.exists() else None,
                'num_cases': len(case_results),
                'total_alphaprun_score': float(total_alphaprun_score),
                'source_cases_file': str(cases_file),
                'results': case_results
            }, f, indent=2)
        
        all_results[model_name] = model_output_file
        print(f"\nSaved results to: {model_output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run scalability benchmarks'
    )
    parser.add_argument('--cases', type=str, required=True,
                       help='Path to scalability cases JSON file')
    parser.add_argument('--scores_dir', type=str, default='../../scores',
                       help='Directory containing score files')
    parser.add_argument('--norms_dir', type=str, default='../weight_norms',
                       help='Directory containing norm files')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    cases_file = Path(args.cases)
    if not cases_file.exists():
        print(f"Error: Cases file not found: {cases_file}")
        return
    
    print("Running scalability benchmarks...")
    print(f"Cases file: {cases_file}")
    print()
    
    results_files = run_scalability_benchmarks(
        cases_file,
        args.scores_dir,
        args.norms_dir,
        args.output_dir
    )
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

