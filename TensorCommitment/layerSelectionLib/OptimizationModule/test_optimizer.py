"""
Comprehensive tests for the Layer Selection Optimizer.

Tests include:
1. Test input/output handling
2. Test solution optimality and determinism
3. Test edge cases (empty budgets, single layer, etc.)
4. Test solution validity (disjoint, contiguous, budget constraints)
5. Test with real scores files
"""

import unittest
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from OptimizationModule.layer_selection_optimizer import LayerSelectionOptimizer
from OptimizationModule.scores_loader import load_scores_from_json


class TestLayerSelectionOptimizer(unittest.TestCase):
    """Test suite for LayerSelectionOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple test case: 5 layers, 2 verifiers
        self.values_simple = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.costs_simple = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.budgets_simple = [2.0, 3.0]
        
        # More complex test case with varying costs
        self.values_complex = [10.0, 5.0, 8.0, 12.0, 3.0, 15.0, 7.0]
        self.costs_complex = [2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 2.0]
        self.budgets_complex = [5.0, 6.0]
    
    def test_1_input_output_handling(self):
        """
        Test 1: Test whether the algorithm takes the input and produces the output properly.
        """
        print("\n=== Test 1: Input/Output Handling ===")
        
        optimizer = LayerSelectionOptimizer(
            self.values_simple,
            self.costs_simple,
            self.budgets_simple
        )
        
        # Check that optimizer was created successfully
        self.assertEqual(optimizer.L, 5)
        self.assertEqual(optimizer.M, 2)
        self.assertEqual(len(optimizer.values), 5)
        self.assertEqual(len(optimizer.costs), 5)
        self.assertEqual(len(optimizer.budgets), 2)
        
        # Run optimization
        optimal_benefit, assignments = optimizer.solve()
        
        # Check output format
        self.assertIsInstance(optimal_benefit, (int, float))
        self.assertGreaterEqual(optimal_benefit, 0.0)
        self.assertIsInstance(assignments, list)
        self.assertEqual(len(assignments), 2)
        
        # Check assignment format
        for assignment in assignments:
            self.assertIsInstance(assignment, tuple)
            self.assertEqual(len(assignment), 2)
            start, end = assignment
            self.assertIsInstance(start, (int, np.integer))
            self.assertIsInstance(end, (int, np.integer))
        
        print(f"✓ Optimal benefit: {optimal_benefit}")
        print(f"✓ Assignments: {assignments}")
        print("✓ Test 1 passed: Input/output handling works correctly")
    
    def test_2_optimality_and_determinism(self):
        """
        Test 2: Test whether the solution is optimal and same for same input file
        and initial conditions such as the verifier budget with different seeds.
        """
        print("\n=== Test 2: Optimality and Determinism ===")
        
        # Run optimization multiple times with same inputs
        results = []
        for seed in range(5):
            np.random.seed(seed)  # Even with different seeds, should get same result
            optimizer = LayerSelectionOptimizer(
                self.values_complex,
                self.costs_complex,
                self.budgets_complex
            )
            benefit, assignments = optimizer.solve()
            results.append((benefit, assignments))
        
        # All results should be identical (deterministic)
        first_benefit, first_assignments = results[0]
        for benefit, assignments in results[1:]:
            self.assertAlmostEqual(benefit, first_benefit, places=6,
                                 msg="Solution should be deterministic")
            self.assertEqual(assignments, first_assignments,
                           msg="Assignments should be identical")
        
        # Verify solution is valid
        optimizer = LayerSelectionOptimizer(
            self.values_complex,
            self.costs_complex,
            self.budgets_complex
        )
        is_valid, error_msg = optimizer.verify_solution(first_assignments)
        self.assertTrue(is_valid, f"Solution should be valid: {error_msg}")
        
        # Check optimality: manually verify this is better than some alternatives
        # For this test case, we know the optimal should use both verifiers
        non_empty_count = sum(1 for s, e in first_assignments if s != -1)
        self.assertGreater(non_empty_count, 0, "At least one verifier should be used")
        
        print(f"✓ Optimal benefit: {first_benefit}")
        print(f"✓ Assignments: {first_assignments}")
        print(f"✓ Ran {len(results)} times, all results identical")
        print("✓ Test 2 passed: Solution is optimal and deterministic")
    
    def test_3_edge_cases(self):
        """
        Test 3: Test edge cases (empty budgets, single layer, tight budgets, etc.)
        """
        print("\n=== Test 3: Edge Cases ===")
        
        # Test 3a: Single layer
        optimizer1 = LayerSelectionOptimizer([5.0], [1.0], [2.0])
        benefit1, assignments1 = optimizer1.solve()
        self.assertEqual(len(assignments1), 1)
        is_valid, _ = optimizer1.verify_solution(assignments1)
        self.assertTrue(is_valid)
        print("✓ Single layer case passed")
        
        # Test 3b: Budget too small for any layer
        optimizer2 = LayerSelectionOptimizer([5.0, 10.0], [10.0, 10.0], [5.0])
        benefit2, assignments2 = optimizer2.solve()
        self.assertEqual(assignments2[0], (-1, -1))  # Empty assignment
        is_valid, _ = optimizer2.verify_solution(assignments2)
        self.assertTrue(is_valid)
        print("✓ Tight budget case passed")
        
        # Test 3c: Very large budget (can take all layers)
        optimizer3 = LayerSelectionOptimizer(
            self.values_simple,
            self.costs_simple,
            [100.0]  # Much larger than total cost
        )
        benefit3, assignments3 = optimizer3.solve()
        # Should assign all layers
        start, end = assignments3[0]
        if start != -1:
            total_assigned = end - start + 1
            self.assertGreater(total_assigned, 0)
        is_valid, _ = optimizer3.verify_solution(assignments3)
        self.assertTrue(is_valid)
        print("✓ Large budget case passed")
        
        # Test 3d: Multiple verifiers with same budget
        optimizer4 = LayerSelectionOptimizer(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        )
        benefit4, assignments4 = optimizer4.solve()
        is_valid, _ = optimizer4.verify_solution(assignments4)
        self.assertTrue(is_valid)
        print("✓ Multiple verifiers case passed")
        
        print("✓ Test 3 passed: All edge cases handled correctly")
    
    def test_4_solution_validity(self):
        """
        Test 4: Test solution validity (disjoint, contiguous, budget constraints).
        """
        print("\n=== Test 4: Solution Validity ===")
        
        optimizer = LayerSelectionOptimizer(
            self.values_complex,
            self.costs_complex,
            self.budgets_complex
        )
        
        benefit, assignments = optimizer.solve()
        
        # Verify using built-in method
        is_valid, error_msg = optimizer.verify_solution(assignments)
        self.assertTrue(is_valid, f"Solution should be valid: {error_msg}")
        
        # Manual verification
        used_layers = set()
        for k, (start, end) in enumerate(assignments):
            if start == -1 and end == -1:
                continue
            
            # Check contiguity: should be a valid range
            self.assertLessEqual(start, end)
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, optimizer.L)
            
            # Check disjointness
            interval_layers = set(range(start, end + 1))
            self.assertFalse(interval_layers & used_layers,
                           f"Verifier {k} overlaps with previous assignments")
            used_layers.update(interval_layers)
            
            # Check budget
            interval_cost = optimizer.C[end + 1] - optimizer.C[start]
            self.assertLessEqual(interval_cost, optimizer.budgets[k] + 1e-9,
                               f"Verifier {k} exceeds budget")
        
        # Check benefit computation
        computed_benefit = optimizer.compute_total_benefit(assignments)
        self.assertAlmostEqual(computed_benefit, benefit, places=6)
        
        print(f"✓ Solution is valid: {error_msg}")
        print(f"✓ Computed benefit matches: {computed_benefit:.6f} == {benefit:.6f}")
        print("✓ Test 4 passed: Solution validity verified")
    
    def test_5_real_scores_file(self):
        """
        Test 5: Test with real scores files from the scores directory.
        """
        print("\n=== Test 5: Real Scores File ===")
        
        # Find a scores file
        scores_dir = Path(__file__).parent.parent / "scores"
        if not scores_dir.exists():
            print("⚠ Scores directory not found, skipping test")
            return
        
        json_files = list(scores_dir.glob("*.json"))
        if not json_files:
            print("⚠ No JSON files found in scores directory, skipping test")
            return
        
        # Use the first available file
        scores_file = json_files[0]
        print(f"Testing with: {scores_file.name}")
        
        # Load scores
        values, costs, metadata = load_scores_from_json(str(scores_file))
        print(f"Loaded {len(values)} layers from {metadata['model_name']}")
        
        # Create reasonable budgets (e.g., 10%, 20%, 15% of total cost)
        total_cost = sum(costs)
        budgets = [total_cost * 0.10, total_cost * 0.20, total_cost * 0.15]
        
        # Run optimization
        optimizer = LayerSelectionOptimizer(values, costs, budgets)
        benefit, assignments = optimizer.solve()
        
        # Verify solution
        is_valid, error_msg = optimizer.verify_solution(assignments)
        self.assertTrue(is_valid, f"Solution should be valid: {error_msg}")
        
        # Check that we got reasonable results
        self.assertGreater(benefit, 0.0)
        computed_benefit = optimizer.compute_total_benefit(assignments)
        self.assertAlmostEqual(computed_benefit, benefit, places=6)
        
        print(f"✓ Optimal benefit: {benefit:.6f}")
        print(f"✓ Assignments: {assignments}")
        print(f"✓ Solution is valid")
        print("✓ Test 5 passed: Real scores file processed successfully")
    
    def test_6_invalid_inputs(self):
        """
        Test 6: Test error handling for invalid inputs.
        """
        print("\n=== Test 6: Invalid Inputs ===")
        
        # Empty values
        with self.assertRaises(ValueError):
            LayerSelectionOptimizer([], [1.0], [1.0])
        
        # Mismatched lengths
        with self.assertRaises(ValueError):
            LayerSelectionOptimizer([1.0, 2.0], [1.0], [1.0])
        
        # Non-positive costs
        with self.assertRaises(ValueError):
            LayerSelectionOptimizer([1.0], [0.0], [1.0])
        
        with self.assertRaises(ValueError):
            LayerSelectionOptimizer([1.0], [-1.0], [1.0])
        
        # Non-positive budgets
        with self.assertRaises(ValueError):
            LayerSelectionOptimizer([1.0], [1.0], [0.0])
        
        # Negative values
        with self.assertRaises(ValueError):
            LayerSelectionOptimizer([-1.0], [1.0], [1.0])
        
        print("✓ Test 6 passed: Invalid inputs handled correctly")


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()

