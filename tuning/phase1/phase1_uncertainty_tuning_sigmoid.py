#!/usr/bin/env python3
"""
Phase 1 Uncertainty Tuning - Sigmoid Optimization Demo
Generates dummy data, runs grid search, then sigmoid optimization
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.stats import pearsonr
from scipy.optimize import minimize

# Import sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ERROR: sklearn is required. Install with: pip install scikit-learn")
    exit(1)


@dataclass
class DummyPosition:
    """Dummy position data for testing"""
    E: float  # Policy cross-entropy
    K: float  # Value sparseness
    stones_on_board: int  # Number of stones
    target: int  # Binary target (0 or 1)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty detection parameters"""
    w1: float  # Weight for policy cross-entropy
    w2: float  # Weight for value sparseness
    a: float   # Phase function slope
    b: float   # Phase function intercept

    def compute_phase(self, stones_on_board: int) -> float:
        """Compute phase multiplier: phase(s) = a*(s/361) + b"""
        return self.a * (stones_on_board / 361.0) + self.b

    def compute_uncertainty(self, E: float, K: float, stones_on_board: int) -> float:
        """Compute uncertainty: (w1*E + w2*K) * phase(s)"""
        phase = self.compute_phase(stones_on_board)
        return (self.w1 * E + self.w2 * K) * phase


def generate_dummy_data(n_samples: int = 1000, seed: int = 42) -> List[DummyPosition]:
    """
    Generate dummy data with randomized E, K, stones, and binary targets.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of DummyPosition objects
    """
    np.random.seed(seed)

    print(f"\nGenerating {n_samples} dummy positions...")

    positions = []

    for i in range(n_samples):
        # Randomize E (policy cross-entropy): typically in range [1.0, 5.0]
        E = np.random.uniform(1.0, 5.0)

        # Randomize K (value sparseness): typically in range [0.0, 0.1]
        K = np.random.uniform(0.0, 0.1)

        # Randomize stones on board: [10, 350]
        stones = np.random.randint(10, 351)

        # Generate target based on a simple rule (you can make this more complex)
        # Rule: High E or high K → more likely to be high-error (target=1)
        # Add some noise to make it non-trivial
        prob_high_error = 0.3 * (E / 5.0) + 0.3 * (K / 0.1) + 0.2 * (stones / 361.0)
        prob_high_error = np.clip(prob_high_error, 0.1, 0.9)  # Keep in reasonable range

        target = 1 if np.random.random() < prob_high_error else 0

        positions.append(DummyPosition(E=E, K=K, stones_on_board=stones, target=target))

    # Print statistics
    targets = [p.target for p in positions]
    print(f"  E range: [{min(p.E for p in positions):.3f}, {max(p.E for p in positions):.3f}]")
    print(f"  K range: [{min(p.K for p in positions):.3f}, {max(p.K for p in positions):.3f}]")
    print(f"  Stones range: [{min(p.stones_on_board for p in positions)}, {max(p.stones_on_board for p in positions)}]")
    print(f"  Target=1 (high-error): {sum(targets)} ({100*np.mean(targets):.1f}%)")
    print(f"  Target=0 (low-error): {len(targets) - sum(targets)} ({100*(1-np.mean(targets)):.1f}%)")

    return positions


def grid_search(positions: List[DummyPosition]) -> List[UncertaintyConfig]:
    """
    Grid search over w1, w2, a, b parameters.

    Args:
        positions: List of training positions

    Returns:
        List of configurations sorted by performance
    """
    print("\n" + "="*80)
    print("GRID SEARCH: Testing parameter combinations")
    print("="*80)

    # Extract arrays
    E = np.array([p.E for p in positions])
    K = np.array([p.K for p in positions])
    S = np.array([p.stones_on_board for p in positions])
    y = np.array([p.target for p in positions])

    # Define grid
    w1_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    a_values = [-0.5, -0.3, 0.0, 0.3, 0.5]
    b_values = [0.75, 0.85, 1.0, 1.15, 1.25]

    print(f"\nGrid dimensions:")
    print(f"  w1 values: {w1_values}")
    print(f"  a values: {a_values}")
    print(f"  b values: {b_values}")
    print(f"  Total combinations: {len(w1_values)} × {len(a_values)} × {len(b_values)} = {len(w1_values)*len(a_values)*len(b_values)}")

    configs = []

    # Grid search
    for w1 in w1_values:
        w2 = 1.0 - w1
        for a in a_values:
            for b in b_values:
                config = UncertaintyConfig(w1=w1, w2=w2, a=a, b=b)

                # Compute uncertainty scores
                phase = a * (S / 361.0) + b
                uncertainty = (w1 * E + w2 * K) * phase

                # Compute correlation with targets (as continuous)
                try:
                    corr, _ = pearsonr(uncertainty, y)
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0

                configs.append({
                    'config': config,
                    'correlation': corr,
                    'uncertainty': uncertainty
                })

    # Sort by correlation
    configs.sort(key=lambda x: x['correlation'], reverse=True)

    # Print top 5
    print(f"\nTop 5 configurations:")
    for i, result in enumerate(configs[:5]):
        cfg = result['config']
        print(f"  {i+1}. w1={cfg.w1:.2f}, w2={cfg.w2:.2f}, a={cfg.a:.2f}, b={cfg.b:.2f} → corr={result['correlation']:.4f}")

    return [c['config'] for c in configs]


def sigmoid_optimization(positions: List[DummyPosition],
                        initial_config: UncertaintyConfig = None) -> UncertaintyConfig:
    """
    Optimize parameters using sigmoid model with interaction terms.

    Args:
        positions: List of training positions
        initial_config: Optional starting configuration (from grid search)

    Returns:
        Optimized configuration
    """
    print("\n" + "="*80)
    print("SIGMOID OPTIMIZATION: Training sigmoid model with interactions")
    print("="*80)

    # Extract arrays
    E = np.array([p.E for p in positions])
    K = np.array([p.K for p in positions])
    S = np.array([p.stones_on_board for p in positions])
    y = np.array([p.target for p in positions])

    print(f"\nTraining on {len(positions)} positions")
    print(f"  E range: [{E.min():.3f}, {E.max():.3f}]")
    print(f"  K range: [{K.min():.3f}, {K.max():.3f}]")
    print(f"  Target distribution: {y.sum()} high-error ({100*y.mean():.1f}%), {len(y)-y.sum()} low-error ({100*(1-y.mean()):.1f}%)")

    # Define objective function for phase parameters
    def objective(phase_params):
        """
        Optimize phase parameters (a, b) for best sigmoid model.
        Returns negative ROC-AUC (to minimize).
        """
        a, b = phase_params

        # Compute phase multiplier
        phase = a * (S / 361.0) + b

        # Create interaction features: E*phase, K*phase
        X_interactions = np.column_stack([
            E * phase,  # Policy cross-entropy weighted by phase
            K * phase   # Value sparseness weighted by phase
        ])

        # Train logistic regression on interaction features
        model = LogisticRegression(
            fit_intercept=True,  # Learn bias term
            max_iter=1000,
            random_state=42
        )

        try:
            model.fit(X_interactions, y)

            # Predict probabilities
            y_pred_proba = model.predict_proba(X_interactions)[:, 1]

            # Compute ROC-AUC (how well we separate high/low error)
            auc = roc_auc_score(y, y_pred_proba)

            # Return negative AUC (minimize to maximize)
            return -auc
        except:
            return 1e6  # Penalty for invalid model

    # Initial guess
    if initial_config is not None:
        x0 = [initial_config.a, initial_config.b]
        print(f"\nStarting from grid search best: a={initial_config.a:.4f}, b={initial_config.b:.4f}")
    else:
        x0 = [0.0, 1.0]
        print(f"\nStarting from default: a=0.0, b=1.0")

    # Bounds
    bounds = [
        (-1.0, 1.0),  # a (slope)
        (0.5, 1.5)    # b (intercept)
    ]

    print("\nOptimizing phase function parameters...")
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

    print(f"\nOptimal phase function: phase(s) = {a_opt:.4f}*(s/361) + {b_opt:.4f}")
    print(f"ROC-AUC achieved: {-result.fun:.4f}")

    # Train final logistic regression model with optimal phase
    X_final = np.column_stack([
        E * phase_opt,
        K * phase_opt
    ])

    final_model = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42)
    final_model.fit(X_final, y)

    # Extract learned weights
    w_E = final_model.coef_[0, 0]  # Weight for E*phase interaction
    w_K = final_model.coef_[0, 1]  # Weight for K*phase interaction
    w_const = final_model.intercept_[0]  # Bias term

    print(f"\nLearned sigmoid model:")
    print(f"  P(high_error) = sigmoid({w_E:.4f}*E*phase + {w_K:.4f}*K*phase + {w_const:.4f})")

    # Convert to w1, w2 format (normalized weights)
    w_total = abs(w_E) + abs(w_K)
    if w_total > 0:
        w1_opt = abs(w_E) / w_total
        w2_opt = abs(w_K) / w_total
    else:
        w1_opt = 0.5
        w2_opt = 0.5

    optimal_config = UncertaintyConfig(
        w1=w1_opt,
        w2=w2_opt,
        a=a_opt,
        b=b_opt
    )

    return optimal_config


def main():
    """Main execution function"""
    print("="*80)
    print("PHASE 1 UNCERTAINTY TUNING - SIGMOID OPTIMIZATION DEMO")
    print("="*80)

    # Step 1: Generate dummy data
    positions = generate_dummy_data(n_samples=1000, seed=42)

    # Step 2: Grid search to find good starting points
    grid_configs = grid_search(positions)
    best_grid_config = grid_configs[0]

    print(f"\n" + "="*80)
    print("BEST GRID SEARCH CONFIGURATION:")
    print(f"  w1 (policy weight): {best_grid_config.w1:.4f}")
    print(f"  w2 (value weight): {best_grid_config.w2:.4f}")
    print(f"  a (phase slope): {best_grid_config.a:.4f}")
    print(f"  b (phase intercept): {best_grid_config.b:.4f}")
    print("="*80)

    # Step 3: Sigmoid optimization starting from best grid config
    sigmoid_config = sigmoid_optimization(positions, initial_config=best_grid_config)

    # Step 4: Print final optimized results
    print("\n" + "="*80)
    print("FINAL OPTIMIZED RESULTS (SIGMOID)")
    print("="*80)
    print("\nOptimized Weights:")
    print(f"  w1 (weight for E - policy cross-entropy): {sigmoid_config.w1:.6f}")
    print(f"  w2 (weight for K - value sparseness):     {sigmoid_config.w2:.6f}")
    print(f"  Sum (should be 1.0):                      {sigmoid_config.w1 + sigmoid_config.w2:.6f}")

    print("\nOptimized Phase Function:")
    print(f"  a (slope):                                {sigmoid_config.a:.6f}")
    print(f"  b (intercept):                            {sigmoid_config.b:.6f}")
    print(f"  Formula: phase(s) = {sigmoid_config.a:.6f} * (s/361) + {sigmoid_config.b:.6f}")

    print("\nPhase Function at Different Game Stages:")
    for stones in [50, 100, 150, 200, 250, 300]:
        phase_val = sigmoid_config.compute_phase(stones)
        print(f"  Stones={stones:3d} → phase={phase_val:.4f}")

    print("\nComplete Uncertainty Formula:")
    print(f"  uncertainty = ({sigmoid_config.w1:.4f}*E + {sigmoid_config.w2:.4f}*K) * ({sigmoid_config.a:.4f}*s/361 + {sigmoid_config.b:.4f})")

    print("\n" + "="*80)
    print("COMPARISON: Grid Search vs Sigmoid Optimization")
    print("="*80)
    print(f"{'Parameter':<20} {'Grid Search':<15} {'Sigmoid Opt':<15} {'Change':<15}")
    print("-"*80)
    print(f"{'w1 (policy)':<20} {best_grid_config.w1:<15.4f} {sigmoid_config.w1:<15.4f} {sigmoid_config.w1 - best_grid_config.w1:+.4f}")
    print(f"{'w2 (value)':<20} {best_grid_config.w2:<15.4f} {sigmoid_config.w2:<15.4f} {sigmoid_config.w2 - best_grid_config.w2:+.4f}")
    print(f"{'a (slope)':<20} {best_grid_config.a:<15.4f} {sigmoid_config.a:<15.4f} {sigmoid_config.a - best_grid_config.a:+.4f}")
    print(f"{'b (intercept)':<20} {best_grid_config.b:<15.4f} {sigmoid_config.b:<15.4f} {sigmoid_config.b - best_grid_config.b:+.4f}")
    print("="*80)

    print("\n✓ Optimization complete!")


if __name__ == "__main__":
    main()
