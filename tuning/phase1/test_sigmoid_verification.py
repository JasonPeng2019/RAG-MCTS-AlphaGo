#!/usr/bin/env python3
"""
Sigmoid Optimization Verification Test
Creates data with KNOWN patterns to verify sigmoid optimization finds the correct thresholds
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.stats import pearsonr
from scipy.optimize import minimize

# Import sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ERROR: sklearn is required. Install with: pip install scikit-learn")
    exit(1)


@dataclass
class TestPosition:
    """Test position with known pattern"""
    E: float  # Policy cross-entropy
    K: float  # Value sparseness
    stones_on_board: int
    target: int  # Binary target (0 or 1)
    true_uncertainty: float  # Ground truth uncertainty score


def generate_test_data_with_pattern(n_samples: int = 1000,
                                     seed: int = 42,
                                     pattern_type: str = "linear") -> Tuple[List[TestPosition], dict]:
    """
    Generate test data with KNOWN patterns.

    Patterns available:
    1. "linear": Simple linear threshold - target=1 if E > threshold_E OR K > threshold_K
    2. "combined": Combined rule - target=1 if (w_true*E + (1-w_true)*K) > threshold
    3. "phase_dependent": Uncertainty depends on game phase

    Returns:
        (positions, ground_truth_params)
    """
    np.random.seed(seed)

    print(f"\n{'='*80}")
    print(f"GENERATING TEST DATA WITH KNOWN PATTERN: {pattern_type}")
    print(f"{'='*80}")

    positions = []

    if pattern_type == "linear":
        # Pattern: target=1 if E > 3.0 OR K > 0.06
        threshold_E = 3.0
        threshold_K = 0.06

        print(f"\nGround Truth Pattern:")
        print(f"  target = 1 if (E > {threshold_E}) OR (K > {threshold_K})")
        print(f"  target = 0 otherwise")

        for i in range(n_samples):
            E = np.random.uniform(1.0, 5.0)
            K = np.random.uniform(0.0, 0.1)
            stones = np.random.randint(10, 351)

            # Deterministic rule
            target = 1 if (E > threshold_E or K > threshold_K) else 0
            true_uncertainty = max((E - threshold_E) / 2.0, (K - threshold_K) / 0.04)

            positions.append(TestPosition(E=E, K=K, stones_on_board=stones,
                                         target=target, true_uncertainty=true_uncertainty))

        ground_truth = {
            'threshold_E': threshold_E,
            'threshold_K': threshold_K,
            'pattern': 'linear'
        }

    elif pattern_type == "combined":
        # Pattern: target=1 if (0.7*E + 0.3*K) > threshold
        # This tests if sigmoid can learn the correct weights
        w_true_E = 0.7
        w_true_K = 0.3
        threshold = 2.5

        print(f"\nGround Truth Pattern:")
        print(f"  uncertainty = {w_true_E}*E + {w_true_K}*K")
        print(f"  target = 1 if uncertainty > {threshold}")
        print(f"  target = 0 otherwise")
        print(f"\nSigmoid should learn: w_E ≈ {w_true_E}, w_K ≈ {w_true_K}")

        for i in range(n_samples):
            E = np.random.uniform(1.0, 5.0)
            K = np.random.uniform(0.0, 0.1)
            stones = np.random.randint(10, 351)

            # Compute true uncertainty
            true_uncertainty = w_true_E * E + w_true_K * K

            # Deterministic rule with some noise
            noise = np.random.normal(0, 0.1)  # Add small noise
            target = 1 if (true_uncertainty + noise) > threshold else 0

            positions.append(TestPosition(E=E, K=K, stones_on_board=stones,
                                         target=target, true_uncertainty=true_uncertainty))

        ground_truth = {
            'w_E': w_true_E,
            'w_K': w_true_K,
            'threshold': threshold,
            'pattern': 'combined'
        }

    elif pattern_type == "phase_dependent":
        # Pattern: Uncertainty increases with game phase
        # target=1 if (E*phase + K*phase) > threshold, where phase = 0.3*(s/361) + 0.8
        w_true_E = 0.6
        w_true_K = 0.4
        a_true = 0.3  # Phase slope
        b_true = 0.8  # Phase intercept
        threshold = 2.0

        print(f"\nGround Truth Pattern:")
        print(f"  phase(s) = {a_true}*(s/361) + {b_true}")
        print(f"  uncertainty = ({w_true_E}*E + {w_true_K}*K) * phase(s)")
        print(f"  target = 1 if uncertainty > {threshold}")
        print(f"\nSigmoid should learn:")
        print(f"  w_E ≈ {w_true_E}, w_K ≈ {w_true_K}")
        print(f"  a ≈ {a_true}, b ≈ {b_true}")

        for i in range(n_samples):
            E = np.random.uniform(1.0, 5.0)
            K = np.random.uniform(0.0, 0.1)
            stones = np.random.randint(10, 351)

            # Compute true phase and uncertainty
            phase = a_true * (stones / 361.0) + b_true
            true_uncertainty = (w_true_E * E + w_true_K * K) * phase

            # Deterministic rule with minimal noise
            noise = np.random.normal(0, 0.05)
            target = 1 if (true_uncertainty + noise) > threshold else 0

            positions.append(TestPosition(E=E, K=K, stones_on_board=stones,
                                         target=target, true_uncertainty=true_uncertainty))

        ground_truth = {
            'w_E': w_true_E,
            'w_K': w_true_K,
            'a': a_true,
            'b': b_true,
            'threshold': threshold,
            'pattern': 'phase_dependent'
        }

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    # Print statistics
    targets = [p.target for p in positions]
    uncertainties = [p.true_uncertainty for p in positions]

    print(f"\nGenerated {n_samples} positions:")
    print(f"  E range: [{min(p.E for p in positions):.3f}, {max(p.E for p in positions):.3f}]")
    print(f"  K range: [{min(p.K for p in positions):.3f}, {max(p.K for p in positions):.3f}]")
    print(f"  Stones range: [{min(p.stones_on_board for p in positions)}, {max(p.stones_on_board for p in positions)}]")
    print(f"  True uncertainty range: [{min(uncertainties):.3f}, {max(uncertainties):.3f}]")
    print(f"  Target=1: {sum(targets)} ({100*np.mean(targets):.1f}%)")
    print(f"  Target=0: {len(targets)-sum(targets)} ({100*(1-np.mean(targets)):.1f}%)")

    return positions, ground_truth


def sigmoid_optimization_test(positions: List[TestPosition]) -> dict:
    """
    Run sigmoid optimization and return learned parameters.
    """
    print(f"\n{'='*80}")
    print("RUNNING SIGMOID OPTIMIZATION")
    print(f"{'='*80}")

    # Extract arrays
    E = np.array([p.E for p in positions])
    K = np.array([p.K for p in positions])
    S = np.array([p.stones_on_board for p in positions])
    y = np.array([p.target for p in positions])

    print(f"\nTraining on {len(positions)} positions")
    print(f"  Target distribution: {y.sum()} high-error ({100*y.mean():.1f}%), {len(y)-y.sum()} low-error ({100*(1-y.mean()):.1f}%)")

    # Define objective function for phase parameters
    def objective(phase_params):
        """Optimize phase parameters (a, b) for best sigmoid model."""
        a, b = phase_params

        # Compute phase multiplier
        phase = a * (S / 361.0) + b

        # Create interaction features: E*phase, K*phase
        X_interactions = np.column_stack([
            E * phase,
            K * phase
        ])

        # Train logistic regression
        model = LogisticRegression(
            fit_intercept=True,
            max_iter=1000,
            random_state=42
        )

        try:
            model.fit(X_interactions, y)
            y_pred_proba = model.predict_proba(X_interactions)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            return -auc
        except:
            return 1e6

    # Optimize phase parameters
    print("\nOptimizing phase function parameters...")

    x0 = [0.0, 1.0]
    bounds = [(-1.0, 1.0), (0.5, 1.5)]

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )

    # Extract optimal phase parameters
    a_opt, b_opt = result.x
    phase_opt = a_opt * (S / 361.0) + b_opt

    print(f"\nOptimized phase function:")
    print(f"  a (slope): {a_opt:.6f}")
    print(f"  b (intercept): {b_opt:.6f}")
    print(f"  ROC-AUC: {-result.fun:.6f}")

    # Train final model
    X_final = np.column_stack([E * phase_opt, K * phase_opt])
    final_model = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42)
    final_model.fit(X_final, y)

    # Extract learned weights
    w_E = final_model.coef_[0, 0]
    w_K = final_model.coef_[0, 1]
    w_const = final_model.intercept_[0]

    # Normalize weights
    w_total = abs(w_E) + abs(w_K)
    w1_norm = abs(w_E) / w_total if w_total > 0 else 0.5
    w2_norm = abs(w_K) / w_total if w_total > 0 else 0.5

    # Compute accuracy
    y_pred = final_model.predict(X_final)
    accuracy = accuracy_score(y, y_pred)

    print(f"\nLearned sigmoid model:")
    print(f"  Raw weights: w_E={w_E:.6f}, w_K={w_K:.6f}, bias={w_const:.6f}")
    print(f"  Normalized: w1={w1_norm:.6f}, w2={w2_norm:.6f}")
    print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.1f}%)")

    return {
        'w_E_raw': w_E,
        'w_K_raw': w_K,
        'w_const': w_const,
        'w1_normalized': w1_norm,
        'w2_normalized': w2_norm,
        'a': a_opt,
        'b': b_opt,
        'roc_auc': -result.fun,
        'accuracy': accuracy
    }


def verify_results(learned_params: dict, ground_truth: dict, pattern_type: str):
    """Compare learned parameters with ground truth."""
    print(f"\n{'='*80}")
    print("VERIFICATION: Learned vs Ground Truth")
    print(f"{'='*80}")

    if pattern_type == "combined":
        w_true_E = ground_truth['w_E']
        w_true_K = ground_truth['w_K']
        w_learned_E = learned_params['w1_normalized']
        w_learned_K = learned_params['w2_normalized']

        error_E = abs(w_learned_E - w_true_E)
        error_K = abs(w_learned_K - w_true_K)

        print(f"\nWeight for E (policy cross-entropy):")
        print(f"  Ground Truth:  {w_true_E:.6f}")
        print(f"  Learned:       {w_learned_E:.6f}")
        print(f"  Error:         {error_E:.6f} ({'✓ PASS' if error_E < 0.1 else '✗ FAIL'})")

        print(f"\nWeight for K (value sparseness):")
        print(f"  Ground Truth:  {w_true_K:.6f}")
        print(f"  Learned:       {w_learned_K:.6f}")
        print(f"  Error:         {error_K:.6f} ({'✓ PASS' if error_K < 0.1 else '✗ FAIL'})")

        # Overall assessment
        total_error = error_E + error_K
        print(f"\n{'='*60}")
        if total_error < 0.15:
            print("✓ VERIFICATION PASSED: Sigmoid found correct weights!")
        elif total_error < 0.3:
            print("⚠ VERIFICATION PARTIAL: Weights are close but not exact")
        else:
            print("✗ VERIFICATION FAILED: Weights are significantly off")
        print(f"{'='*60}")

    elif pattern_type == "phase_dependent":
        w_true_E = ground_truth['w_E']
        w_true_K = ground_truth['w_K']
        a_true = ground_truth['a']
        b_true = ground_truth['b']

        w_learned_E = learned_params['w1_normalized']
        w_learned_K = learned_params['w2_normalized']
        a_learned = learned_params['a']
        b_learned = learned_params['b']

        error_E = abs(w_learned_E - w_true_E)
        error_K = abs(w_learned_K - w_true_K)
        error_a = abs(a_learned - a_true)
        error_b = abs(b_learned - b_true)

        print(f"\nWeight for E:")
        print(f"  Ground Truth:  {w_true_E:.6f}")
        print(f"  Learned:       {w_learned_E:.6f}")
        print(f"  Error:         {error_E:.6f} ({'✓' if error_E < 0.1 else '✗'})")

        print(f"\nWeight for K:")
        print(f"  Ground Truth:  {w_true_K:.6f}")
        print(f"  Learned:       {w_learned_K:.6f}")
        print(f"  Error:         {error_K:.6f} ({'✓' if error_K < 0.1 else '✗'})")

        print(f"\nPhase slope (a):")
        print(f"  Ground Truth:  {a_true:.6f}")
        print(f"  Learned:       {a_learned:.6f}")
        print(f"  Error:         {error_a:.6f} ({'✓' if error_a < 0.1 else '✗'})")

        print(f"\nPhase intercept (b):")
        print(f"  Ground Truth:  {b_true:.6f}")
        print(f"  Learned:       {b_learned:.6f}")
        print(f"  Error:         {error_b:.6f} ({'✓' if error_b < 0.1 else '✗'})")

        total_error = error_E + error_K + error_a + error_b
        print(f"\n{'='*60}")
        if total_error < 0.3:
            print("✓ VERIFICATION PASSED: Sigmoid found correct parameters!")
        elif total_error < 0.6:
            print("⚠ VERIFICATION PARTIAL: Parameters are close but not exact")
        else:
            print("✗ VERIFICATION FAILED: Parameters are significantly off")
        print(f"{'='*60}")

    print(f"\nModel Performance:")
    print(f"  ROC-AUC:  {learned_params['roc_auc']:.4f} (should be > 0.9 for good recovery)")
    print(f"  Accuracy: {learned_params['accuracy']:.4f} (should be > 0.85 for good recovery)")


def main():
    """Run verification tests with different patterns."""

    print("="*80)
    print("SIGMOID OPTIMIZATION VERIFICATION TEST")
    print("="*80)

    # Test 1: Combined pattern (most relevant for your use case)
    print("\n" + "="*80)
    print("TEST 1: COMBINED PATTERN (w_E=0.7, w_K=0.3)")
    print("="*80)

    positions_combined, ground_truth_combined = generate_test_data_with_pattern(
        n_samples=1000,
        seed=42,
        pattern_type="combined"
    )

    learned_combined = sigmoid_optimization_test(positions_combined)
    verify_results(learned_combined, ground_truth_combined, "combined")

    # Test 2: Phase-dependent pattern
    print("\n" + "="*80)
    print("TEST 2: PHASE-DEPENDENT PATTERN (w_E=0.6, w_K=0.4, a=0.3, b=0.8)")
    print("="*80)

    positions_phase, ground_truth_phase = generate_test_data_with_pattern(
        n_samples=1000,
        seed=123,
        pattern_type="phase_dependent"
    )

    learned_phase = sigmoid_optimization_test(positions_phase)
    verify_results(learned_phase, ground_truth_phase, "phase_dependent")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
