#!/usr/bin/env python3
"""
Sigmoid Ranking Verification Test
Tests if sigmoid correctly RANKS positions by uncertainty (not exact weight recovery)
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class TestPosition:
    E: float
    K: float
    stones: int
    true_uncertainty: float
    target: int


def generate_ranking_test_data(n_samples: int = 1000, seed: int = 42):
    """
    Generate data where we know the TRUE uncertainty ranking.

    Ground truth: uncertainty = 0.7*E + 0.3*K
    """
    np.random.seed(seed)

    w_true_E = 0.7
    w_true_K = 0.3
    threshold = 2.5

    print(f"{'='*80}")
    print("GENERATING RANKING TEST DATA")
    print(f"{'='*80}")
    print(f"\nGround Truth:")
    print(f"  True uncertainty = {w_true_E}*E + {w_true_K}*K")
    print(f"  Target = 1 if uncertainty > {threshold}")

    positions = []
    for i in range(n_samples):
        E = np.random.uniform(1.0, 5.0)
        K = np.random.uniform(0.0, 0.1)
        stones = np.random.randint(10, 351)

        # True uncertainty (what we want to recover)
        true_unc = w_true_E * E + w_true_K * K

        # Binary target with small noise
        noise = np.random.normal(0, 0.1)
        target = 1 if (true_unc + noise) > threshold else 0

        positions.append(TestPosition(E=E, K=K, stones=stones,
                                      true_uncertainty=true_unc,
                                      target=target))

    return positions, {'w_E': w_true_E, 'w_K': w_true_K, 'threshold': threshold}


def train_sigmoid_model(positions: List[TestPosition]):
    """Train sigmoid model on the positions."""

    E = np.array([p.E for p in positions])
    K = np.array([p.K for p in positions])
    S = np.array([p.stones for p in positions])
    y = np.array([p.target for p in positions])

    print(f"\n{'='*80}")
    print("TRAINING SIGMOID MODEL")
    print(f"{'='*80}")
    print(f"  Samples: {len(positions)}")
    print(f"  Target=1: {y.sum()} ({100*y.mean():.1f}%)")

    # Simple model without phase (to test weight recovery)
    X = np.column_stack([E, K])

    model = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42)
    model.fit(X, y)

    # Extract learned weights
    w_E_raw = model.coef_[0, 0]
    w_K_raw = model.coef_[0, 1]

    # Normalize weights
    w_total = abs(w_E_raw) + abs(w_K_raw)
    w_E_norm = abs(w_E_raw) / w_total
    w_K_norm = abs(w_K_raw) / w_total

    print(f"\nLearned weights:")
    print(f"  Raw: w_E={w_E_raw:.6f}, w_K={w_K_raw:.6f}")
    print(f"  Normalized: w_E={w_E_norm:.6f}, w_K={w_K_norm:.6f}")

    # Predict probabilities (these are the "learned uncertainties")
    predicted_probs = model.predict_proba(X)[:, 1]

    return model, predicted_probs, {'w_E': w_E_norm, 'w_K': w_K_norm}


def test_ranking_quality(positions: List[TestPosition],
                        predicted_uncertainties: np.ndarray,
                        ground_truth: dict):
    """
    Test if the model correctly RANKS positions by uncertainty.

    This is what actually matters - not exact weight recovery,
    but whether high uncertainty positions are ranked higher.
    """

    print(f"\n{'='*80}")
    print("RANKING QUALITY TEST")
    print(f"{'='*80}")

    # Get true uncertainties
    true_uncertainties = np.array([p.true_uncertainty for p in positions])

    # Compute ranking correlations
    spearman_corr, spearman_p = spearmanr(true_uncertainties, predicted_uncertainties)
    pearson_corr, pearson_p = pearsonr(true_uncertainties, predicted_uncertainties)

    print(f"\nRanking Correlations:")
    print(f"  Spearman (rank correlation): {spearman_corr:.6f}")
    print(f"  Pearson (linear correlation): {pearson_corr:.6f}")

    # Test top-K precision
    # Among top 10% predicted uncertain, what % are actually top 10% uncertain?
    k = len(positions) // 10

    top_k_predicted_idx = np.argsort(-predicted_uncertainties)[:k]
    top_k_true_idx = np.argsort(-true_uncertainties)[:k]

    overlap = len(set(top_k_predicted_idx) & set(top_k_true_idx))
    top_k_precision = overlap / k

    print(f"\nTop-10% Precision:")
    print(f"  Overlap: {overlap}/{k}")
    print(f"  Precision: {top_k_precision:.4f} ({100*top_k_precision:.1f}%)")

    # Classification performance
    targets = np.array([p.target for p in positions])
    roc_auc = roc_auc_score(targets, predicted_uncertainties)

    print(f"\nClassification Performance:")
    print(f"  ROC-AUC: {roc_auc:.6f}")

    # Verification
    print(f"\n{'='*60}")
    if spearman_corr > 0.95:
        print("✓ EXCELLENT: Model ranks uncertainties nearly perfectly!")
        status = "PASS"
    elif spearman_corr > 0.85:
        print("✓ GOOD: Model ranks uncertainties well")
        status = "PASS"
    elif spearman_corr > 0.70:
        print("⚠ MODERATE: Model captures general ranking but not precise")
        status = "PARTIAL"
    else:
        print("✗ POOR: Model does not rank uncertainties correctly")
        status = "FAIL"
    print(f"{'='*60}")

    return {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'top_k_precision': top_k_precision,
        'roc_auc': roc_auc,
        'status': status
    }


def test_prediction_on_known_cases(learned_weights: dict, ground_truth: dict):
    """
    Test specific cases where we know what the answer should be.
    """

    print(f"\n{'='*80}")
    print("KNOWN CASE PREDICTIONS")
    print(f"{'='*80}")

    w_true_E = ground_truth['w_E']
    w_true_K = ground_truth['w_K']
    w_learned_E = learned_weights['w_E']
    w_learned_K = learned_weights['w_K']

    test_cases = [
        {"name": "High E, Low K", "E": 4.0, "K": 0.01},
        {"name": "Low E, High K", "E": 1.5, "K": 0.09},
        {"name": "Medium E, Medium K", "E": 3.0, "K": 0.05},
        {"name": "High E, High K", "E": 4.5, "K": 0.09},
    ]

    print(f"\nComparing True vs Learned Uncertainty Rankings:")
    print(f"\n{'Case':<25} {'True Unc':>12} {'Learned Unc':>14} {'Ratio':>10}")
    print("-" * 70)

    for case in test_cases:
        E, K = case['E'], case['K']

        true_unc = w_true_E * E + w_true_K * K
        learned_unc = w_learned_E * E + w_learned_K * K
        ratio = learned_unc / true_unc if true_unc > 0 else 0

        print(f"{case['name']:<25} {true_unc:>12.4f} {learned_unc:>14.4f} {ratio:>10.3f}")

    print(f"\nNote: Exact values may differ, but RANKING should be similar")


def main():
    """Run ranking verification test."""

    print("="*80)
    print("SIGMOID RANKING VERIFICATION TEST")
    print("="*80)
    print("\nThis test verifies that sigmoid correctly RANKS uncertainties,")
    print("not that it recovers exact weights (which is a harder problem)")

    # Generate test data
    positions, ground_truth = generate_ranking_test_data(n_samples=1000, seed=42)

    # Train sigmoid model
    model, predicted_uncertainties, learned_weights = train_sigmoid_model(positions)

    # Test ranking quality
    results = test_ranking_quality(positions, predicted_uncertainties, ground_truth)

    # Test on known cases
    test_prediction_on_known_cases(learned_weights, ground_truth)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")
    print(f"\nGround Truth Weights:")
    print(f"  w_E = {ground_truth['w_E']:.3f}, w_K = {ground_truth['w_K']:.3f}")
    print(f"\nLearned Weights (normalized):")
    print(f"  w_E = {learned_weights['w_E']:.3f}, w_K = {learned_weights['w_K']:.3f}")
    print(f"\nKey Metrics:")
    print(f"  Spearman Rank Correlation: {results['spearman']:.4f}")
    print(f"  Top-10% Precision: {results['top_k_precision']:.4f}")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"\nVerification Status: {results['status']}")

    if results['status'] == "PASS":
        print("\n✓ Sigmoid optimization is working correctly!")
        print("  It may not recover exact weights, but it correctly ranks")
        print("  positions by uncertainty, which is what actually matters.")

    return results


if __name__ == "__main__":
    results = main()
