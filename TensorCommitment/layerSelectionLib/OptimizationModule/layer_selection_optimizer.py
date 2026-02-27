"""
Optimal O(ML) Dynamic Programming Solver for Multi-Verifier Budgeted Contiguous Selection.

Implements the DP algorithm described in the mathematical formulation with:
- O(ML) time complexity using sliding window optimization
- O(L) space complexity per verifier (or O(L) overall with rolling arrays)
- Optimal solution guarantee
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class LayerSelectionOptimizer:
    """
    Solves the Multi-Verifier Budgeted Contiguous Selection (MV-BCS) problem.
    
    Given L layers with benefits v_i and costs c_i, and M verifiers with budgets B_k,
    selects M pairwise-disjoint contiguous intervals (one per verifier) maximizing
    total benefit subject to budget constraints.
    """
    
    def __init__(self, values: List[float], costs: List[float], budgets: List[float]):
        """
        Initialize the optimizer.
        
        Args:
            values: List of L benefit values (v_i >= 0)
            costs: List of L cost values (c_i > 0)
            budgets: List of M verifier budgets (B_k > 0)
        
        Raises:
            ValueError: If inputs are invalid (empty, mismatched lengths, non-positive costs/budgets)
        """
        if not values or not costs or not budgets:
            raise ValueError("Values, costs, and budgets must be non-empty")
        if len(values) != len(costs):
            raise ValueError(f"Values length ({len(values)}) must match costs length ({len(costs)})")
        if any(c <= 0 for c in costs):
            raise ValueError("All costs must be positive")
        if any(b <= 0 for b in budgets):
            raise ValueError("All budgets must be positive")
        if any(v < 0 for v in values):
            raise ValueError("All values must be non-negative")
        
        self.L = len(values)
        self.M = len(budgets)
        self.values = np.array(values, dtype=np.float64)
        self.costs = np.array(costs, dtype=np.float64)
        self.budgets = np.array(budgets, dtype=np.float64)
        
        # Compute prefix sums: V[i] = sum(v_1..v_i), C[i] = sum(c_1..c_i)
        # V[0] = C[0] = 0
        self.V = np.zeros(self.L + 1, dtype=np.float64)
        self.C = np.zeros(self.L + 1, dtype=np.float64)
        self.V[1:] = np.cumsum(self.values)
        self.C[1:] = np.cumsum(self.costs)
        
        # DP table: DP[k, i] = max benefit using verifiers 1..k on layers 1..i
        # DP[0, i] = DP[k, 0] = 0
        self.DP = None
        self.backtrack = None  # For reconstructing solution
    
    def solve(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Solve the optimization problem.
        
        Returns:
            Tuple of (optimal_total_benefit, assignments) where assignments is
            a list of M tuples (start, end) for each verifier's interval.
            Empty intervals are represented as (-1, -1).
        """
        # Initialize DP table: DP[k, i] for k in [0..M], i in [0..L]
        # Use 0-indexed: DP[k][i] = max benefit using verifiers 0..k-1 on layers 0..i-1
        self.DP = np.zeros((self.M + 1, self.L + 1), dtype=np.float64)
        self.backtrack = np.full((self.M + 1, self.L + 1), -1, dtype=np.int32)
        
        # Fill DP table
        for k in range(1, self.M + 1):
            budget_k = self.budgets[k - 1]
            
            # Maintain sliding window deque for amortized O(1) lookup
            # Window W_k(i) = {r in [0, i-1] : C[i] - C[r] <= budget_k}
            # We want to minimize key(r) = V[r] - DP[k-1][r] over feasible r
            # Deque stores indices r sorted by key value (increasing), front has minimum
            window_deque = deque()  # Stores indices r
            
            # For each position i (1-indexed in problem, 0-indexed in code)
            for i in range(1, self.L + 1):
                # Option 1: Don't end any interval at i
                best_value = self.DP[k][i - 1]
                best_start = -1
                
                # Option 2: End verifier k's interval at i
                # Find feasible start positions s such that C[i] - C[s-1] <= budget_k
                # This means C[s-1] >= C[i] - budget_k
                # We want to minimize V[s-1] - DP[k-1][s-1] over feasible s
                # Let r = s-1, so we minimize V[r] - DP[k-1][r] over feasible r
                
                # Update sliding window: remove indices that are no longer feasible
                # Feasible r must satisfy: C[i] - C[r] <= budget_k, i.e., C[r] >= C[i] - budget_k
                min_cost = self.C[i] - budget_k
                
                # Remove indices from front that are too small (no longer feasible)
                while window_deque and self.C[window_deque[0]] < min_cost:
                    window_deque.popleft()
                
                # Add new candidate: r = i-1 (which corresponds to start s = i)
                # This candidate becomes available as i increases
                r_candidate = i - 1
                if r_candidate >= 0:
                    # Check feasibility: C[i] - C[r_candidate] <= budget_k
                    if self.C[i] - self.C[r_candidate] <= budget_k:
                        # Key value for this candidate: we want to minimize this
                        key_val = self.V[r_candidate] - self.DP[k - 1][r_candidate]
                        
                        # Remove indices from back that have worse (larger) key values
                        # We maintain deque in increasing order of key value
                        while window_deque:
                            last_r = window_deque[-1]
                            last_key = self.V[last_r] - self.DP[k - 1][last_r]
                            if last_key >= key_val:
                                window_deque.pop()
                            else:
                                break
                        
                        window_deque.append(r_candidate)
                
                # Now find best start from window (front has minimum key)
                if window_deque:
                    best_r = window_deque[0]
                    # Verify it's still feasible (should always be true after cleanup)
                    if self.C[i] - self.C[best_r] <= budget_k:
                        # Benefit of interval [best_r+1, i] (1-indexed) = V[i] - V[best_r]
                        interval_benefit = self.V[i] - self.V[best_r]
                        total_value = interval_benefit + self.DP[k - 1][best_r]
                        
                        if total_value > best_value:
                            best_value = total_value
                            best_start = best_r + 1  # Convert to 1-indexed start
                
                self.DP[k][i] = best_value
                self.backtrack[k][i] = best_start
        
        # Reconstruct solution
        assignments = self._reconstruct_solution()
        optimal_benefit = self.DP[self.M][self.L]
        
        return optimal_benefit, assignments
    
    def _reconstruct_solution(self) -> List[Tuple[int, int]]:
        """
        Reconstruct the optimal solution by backtracking.
        
        Returns:
            List of M tuples (start, end) for each verifier's interval.
            Empty intervals are (-1, -1). Indices are 0-based.
        """
        assignments = []
        k = self.M
        i = self.L
        
        while k > 0:
            # Check if verifier k is used: DP[k][i] > DP[k-1][i]
            if abs(self.DP[k][i] - self.DP[k-1][i]) < 1e-9:
                # Verifier k is unused
                assignments.append((-1, -1))
                # Don't change i, continue with k-1
            else:
                # Verifier k is used, find where its interval ends
                # Find the last position j <= i where an interval ends (backtrack[k][j] != -1)
                # or where DP value changes (DP[k][j] != DP[k][j-1])
                j = i
                while j > 0:
                    start = self.backtrack[k][j]
                    if start != -1:
                        # Found interval ending at j
                        assignments.append((start - 1, j - 1))
                        i = start - 1
                        break
                    elif j > 0 and abs(self.DP[k][j] - self.DP[k][j-1]) > 1e-9:
                        # DP value changes at j, so interval ends at j
                        # But backtrack is -1, which shouldn't happen. Try to find start manually.
                        # Actually, if backtrack is -1 but DP changes, it means we took DP[k][j-1]
                        # So the interval ends at j-1. But that's also wrong.
                        # Let's check if we can find the start by checking what gives this benefit
                        # For now, skip this case and continue
                        j -= 1
                    else:
                        j -= 1
                else:
                    # No interval found, verifier k is unused
                    assignments.append((-1, -1))
            
            k -= 1
        
        # Reverse to get assignments for verifiers 1..M
        assignments.reverse()
        return assignments
    
    def get_dp_table(self) -> np.ndarray:
        """Return the DP table for inspection (optional)."""
        return self.DP.copy()
    
    def verify_solution(self, assignments: List[Tuple[int, int]]) -> Tuple[bool, str]:
        """
        Verify that a solution is valid (disjoint, within budgets, contiguous).
        
        Args:
            assignments: List of M tuples (start, end) for each verifier
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(assignments) != self.M:
            return False, f"Expected {self.M} assignments, got {len(assignments)}"
        
        used_layers = set()
        
        for k, (start, end) in enumerate(assignments):
            if start == -1 and end == -1:
                continue  # Empty interval is valid
            
            if start < 0 or end < 0 or start > end or end >= self.L:
                return False, f"Verifier {k}: Invalid interval [{start}, {end}]"
            
            # Check contiguity (already guaranteed by construction)
            interval_layers = set(range(start, end + 1))
            
            # Check disjointness
            if interval_layers & used_layers:
                return False, f"Verifier {k}: Interval [{start}, {end}] overlaps with previous assignments"
            
            used_layers.update(interval_layers)
            
            # Check budget
            interval_cost = self.C[end + 1] - self.C[start]
            if interval_cost > self.budgets[k] + 1e-9:  # Small tolerance for floating point
                return False, f"Verifier {k}: Interval [{start}, {end}] cost {interval_cost:.6f} exceeds budget {self.budgets[k]:.6f}"
        
        return True, "Solution is valid"
    
    def compute_total_benefit(self, assignments: List[Tuple[int, int]]) -> float:
        """
        Compute total benefit of a solution.
        
        Args:
            assignments: List of M tuples (start, end) for each verifier
        
        Returns:
            Total benefit
        """
        total = 0.0
        for start, end in assignments:
            if start != -1 and end != -1:
                total += self.V[end + 1] - self.V[start]
        return total

