#!/usr/bin/env python3
"""
Phase 1 Uncertainty Tuning - Optimize Cosine Similarity Prediction (LOG-SCALED VERSION)

This script optimizes w1, w2, and phase function type to predict
the cosine similarity between shallow and deep search results.

LOG-SCALING APPROACH:
- Transforms cosine similarity using log(1 + cos_sim + 1) to reduce influence of outliers
- This gives more even distribution across the y-axis
- Predictions are transformed back to original scale for evaluation

Approach:
1. Grid search over w1 and different phase functions to find best combination
2. Linear regression to optimize weights for predicting LOG-SCALED cosine similarity
3. Target: Cosine similarity between (shallow_value, shallow_prior) and (deep_value, deep_prior)

Phase functions tested:
- s (linear)
- 1/s (inverse)
- s^2 (quadratic)
- 1/(s^2) (inverse quadratic)
- e^(s) (exponential growth)
- e^(-s) (exponential decay)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


@dataclass
class PositionData:
    """Data extracted from a flagged position"""
    game_id: str
    query_id: str
    move_number: int
    stones_on_board: int

    # Uncertainty metrics
    E: float  # policy_entropy
    K: float  # value_variance

    # Shallow search
    shallow_best_move: str
    shallow_value: float
    shallow_prior: float

    # Deep search
    deep_best_move: str
    deep_value: float
    deep_prior: float

    # Target variables
    cosine_similarity: float  # Cosine similarity between shallow and deep
    move_agreement: int  # 1 if moves agree (same), 0 if moves disagree (different)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty formula"""
    w1: float  # Weight for policy entropy (E)
    w2: float  # Weight for value variance (K)
    phase_function_name: str  # Name of phase function

    def __str__(self):
        return f"w1={self.w1:.4f}, w2={self.w2:.4f}, phase={self.phase_function_name}"


# Define phase functions
def phase_linear(s: np.ndarray) -> np.ndarray:
    """Phase = s/361"""
    return s / 361.0


def phase_inverse(s: np.ndarray) -> np.ndarray:
    """Phase = 1/s (handle s=0 case)"""
    result = np.ones_like(s, dtype=float)
    mask = s > 0
    result[mask] = 1.0 / s[mask]
    result[~mask] = 10.0  # Large value for s=0
    return result


def phase_quadratic(s: np.ndarray) -> np.ndarray:
    """Phase = (s/361)^2"""
    return (s / 361.0) ** 2


def phase_inverse_quadratic(s: np.ndarray) -> np.ndarray:
    """Phase = 1/(s^2) (handle s=0 case)"""
    result = np.ones_like(s, dtype=float)
    mask = s > 0
    result[mask] = 1.0 / (s[mask] ** 2)
    result[~mask] = 100.0  # Large value for s=0
    return result


def phase_exp_growth(s: np.ndarray) -> np.ndarray:
    """Phase = e^(s/361) - 1"""
    return np.exp(s / 361.0) - 1.0


def phase_exp_decay(s: np.ndarray) -> np.ndarray:
    """Phase = e^(-s/361)"""
    return np.exp(-s / 361.0)


# Dictionary of phase functions
PHASE_FUNCTIONS = {
    's': phase_linear,
    '1/s': phase_inverse,
    's^2': phase_quadratic,
    '1/(s^2)': phase_inverse_quadratic,
    'e^(s)': phase_exp_growth,
    'e^(-s)': phase_exp_decay
}


def log_transform_cosine(cos_sim: np.ndarray) -> np.ndarray:
    """
    Transform cosine similarity to log scale to reduce influence of outliers.

    Since cosine similarity can be in [-1, 1], we shift it to [0, 2] first,
    then apply log(1 + x) transformation.

    Args:
        cos_sim: Cosine similarity values in range [-1, 1]

    Returns:
        Log-transformed values
    """
    # Shift to [0, 2] range
    shifted = cos_sim + 1.0
    # Apply log transformation: log(1 + x)
    return np.log1p(shifted)


def inverse_log_transform_cosine(log_cos_sim: np.ndarray) -> np.ndarray:
    """
    Inverse transform from log scale back to original cosine similarity scale.

    Args:
        log_cos_sim: Log-transformed cosine similarity values

    Returns:
        Original cosine similarity values in range [-1, 1]
    """
    # Inverse of log(1 + x) is exp(y) - 1
    shifted = np.expm1(log_cos_sim)
    # Shift back to [-1, 1] range
    return shifted - 1.0


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector [value, prior]
        v2: Second vector [value, prior]

    Returns:
        Cosine similarity in range [-1, 1]
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def load_rag_data(rag_data_dir: Path) -> List[PositionData]:
    """
    Load all RAG data from JSON files and compute cosine similarity.

    Returns:
        List of PositionData objects
    """
    all_positions = []
    json_files = list(rag_data_dir.glob("RAG_rawdata_game_*.json"))

    print(f"Loading data from {len(json_files)} JSON files...")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            game_id = data.get("game_id", "")
            flagged_positions = data.get("flagged_positions", [])

            for flagged_pos in flagged_positions:
                # Extract uncertainty metrics
                uncertainty = flagged_pos.get("uncertainty_metrics", {})
                E = uncertainty.get("policy_entropy")
                K = uncertainty.get("value_variance")

                if E is None or K is None:
                    continue

                # Extract shallow search data
                shallow_best_move = flagged_pos.get("best_move", {}).get("move")
                shallow_children = flagged_pos.get("children", [])

                shallow_value = None
                shallow_prior = None
                for child in shallow_children:
                    if child.get("move") == shallow_best_move:
                        shallow_value = child.get("value")
                        shallow_prior = child.get("prior")
                        break

                # Extract deep search data
                deep_result = flagged_pos.get("deep_result", {})

                if not deep_result.get("available") or deep_result.get("status") != "ok":
                    continue

                deep_best_move = deep_result.get("best_move", {}).get("move")
                deep_children = deep_result.get("children", [])

                deep_value = None
                deep_prior = None
                for child in deep_children:
                    if child.get("move") == deep_best_move:
                        deep_value = child.get("value")
                        deep_prior = child.get("prior")
                        break

                # Check if we have all required data
                if (shallow_value is None or shallow_prior is None or
                    deep_value is None or deep_prior is None):
                    continue

                # Compute cosine similarity
                shallow_vec = np.array([shallow_value, shallow_prior])
                deep_vec = np.array([deep_value, deep_prior])
                cos_sim = cosine_similarity(shallow_vec, deep_vec)

                # Determine move agreement: 1 if same, 0 if different
                move_agreement = 1 if shallow_best_move == deep_best_move else 0

                # Create position data
                pos = PositionData(
                    game_id=game_id,
                    query_id=flagged_pos.get("query_id", ""),
                    move_number=flagged_pos.get("move_number", 0),
                    stones_on_board=flagged_pos.get("stone_count", {}).get("total", 0),
                    E=E,
                    K=K,
                    shallow_best_move=shallow_best_move,
                    shallow_value=shallow_value,
                    shallow_prior=shallow_prior,
                    deep_best_move=deep_best_move,
                    deep_value=deep_value,
                    deep_prior=deep_prior,
                    cosine_similarity=cos_sim,
                    move_agreement=move_agreement
                )
                all_positions.append(pos)

        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")
            continue

    return all_positions


def grid_search(positions_train: List[PositionData]) -> Tuple[UncertaintyConfig, float, str]:
    """
    Grid search over w1 and different phase functions.
    Goal: Find parameters where uncertainty correlates with LOW cosine similarity
    (high uncertainty → shallow and deep disagree more)

    Returns:
        Best configuration, correlation score, and phase function name
    """
    print("\n" + "="*80)
    print("GRID SEARCH: Testing w1 weights × phase functions")
    print("="*80)

    # Extract arrays
    E = np.array([p.E for p in positions_train])
    K = np.array([p.K for p in positions_train])
    S = np.array([p.stones_on_board for p in positions_train])
    cos_sim = np.array([p.cosine_similarity for p in positions_train])

    # Apply log transformation to cosine similarity
    log_cos_sim = log_transform_cosine(cos_sim)

    print(f"\nTraining set: {len(positions_train)} positions")
    print(f"  E range: [{E.min():.3f}, {E.max():.3f}]")
    print(f"  K range: [{K.min():.6f}, {K.max():.6f}]")
    print(f"  Stones range: [{S.min()}, {S.max()}]")
    print(f"  Cosine similarity range: [{cos_sim.min():.4f}, {cos_sim.max():.4f}]")
    print(f"  Cosine similarity mean: {cos_sim.mean():.4f}")
    print(f"  LOG-TRANSFORMED cos_sim range: [{log_cos_sim.min():.4f}, {log_cos_sim.max():.4f}]")
    print(f"  LOG-TRANSFORMED cos_sim mean: {log_cos_sim.mean():.4f}")

    # Define grid
    w1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    total_configs = len(w1_values) * len(PHASE_FUNCTIONS)

    print(f"\nGrid dimensions:")
    print(f"  w1 values: {len(w1_values)} → {w1_values}")
    print(f"  Phase functions: {len(PHASE_FUNCTIONS)} → {list(PHASE_FUNCTIONS.keys())}")
    print(f"  Total combinations: {total_configs}")

    best_config = None
    best_score = -float('inf')
    all_results = []

    # Test each phase function
    for phase_name, phase_func in PHASE_FUNCTIONS.items():
        print(f"\n  → Testing phase function: {phase_name}")

        # Compute phase values
        phase = phase_func(S)

        # Test each w1 value
        for w1 in w1_values:
            w2 = 1.0 - w1
            config = UncertaintyConfig(w1=w1, w2=w2, phase_function_name=phase_name)

            # Compute uncertainty scores
            uncertainty = (w1 * E + w2 * K) * phase

            # High uncertainty should correlate with LOW cosine similarity (or LOW log-cosine)
            # So we want negative correlation
            try:
                corr, _ = pearsonr(-uncertainty, log_cos_sim)
                if np.isnan(corr):
                    corr = 0.0
            except:
                corr = 0.0

            all_results.append({
                'config': config,
                'correlation': corr,
                'phase_name': phase_name
            })

            if corr > best_score:
                best_score = corr
                best_config = config

    # Sort by correlation
    all_results.sort(key=lambda x: x['correlation'], reverse=True)

    # Print top 15
    print(f"\n" + "-"*80)
    print("TOP 15 CONFIGURATIONS (across all phase functions):")
    print("-"*80)
    for i, result in enumerate(all_results[:15]):
        cfg = result['config']
        print(f"  {i+1:2d}. w1={cfg.w1:.4f}, w2={cfg.w2:.4f}, phase={cfg.phase_function_name:<10s} → corr={result['correlation']:.4f}")

    # Show best for each phase function
    print(f"\n" + "-"*80)
    print("BEST CONFIGURATION PER PHASE FUNCTION:")
    print("-"*80)
    for phase_name in PHASE_FUNCTIONS.keys():
        phase_results = [r for r in all_results if r['phase_name'] == phase_name]
        if phase_results:
            best_for_phase = phase_results[0]
            cfg = best_for_phase['config']
            print(f"  {phase_name:<10s}: w1={cfg.w1:.4f}, w2={cfg.w2:.4f} → corr={best_for_phase['correlation']:.4f}")

    print(f"\n" + "="*80)
    print(f"✓ OVERALL BEST CONFIG FROM GRID SEARCH: {best_config}")
    print(f"  Correlation: {best_score:.4f}")
    print(f"  ✓ SELECTED PHASE FUNCTION: {best_config.phase_function_name}")
    print("="*80)

    return best_config, best_score, best_config.phase_function_name


def linear_regression_optimization(
    positions_train: List[PositionData],
    phase_function_name: str,
    initial_w1: float = 0.5
) -> Tuple[UncertaintyConfig, LinearRegression, Dict]:
    """
    Optimize weights using linear regression with a fixed phase function.

    The model predicts cosine similarity from uncertainty features.
    We use the best phase function from grid search and learn optimal w1, w2 weights.

    Returns:
        Tuple of (optimized config, trained model, evaluation metrics)
    """
    print("\n" + "="*80)
    print("LINEAR REGRESSION OPTIMIZATION")
    print("="*80)

    # Extract arrays
    E = np.array([p.E for p in positions_train])
    K = np.array([p.K for p in positions_train])
    S = np.array([p.stones_on_board for p in positions_train])
    cos_sim = np.array([p.cosine_similarity for p in positions_train])

    # Apply log transformation to target variable
    log_cos_sim = log_transform_cosine(cos_sim)

    print(f"\nTraining on {len(positions_train)} positions")
    print(f"  Target: Cosine similarity (mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f})")
    print(f"  LOG-TRANSFORMED target (mean={log_cos_sim.mean():.4f}, std={log_cos_sim.std():.4f})")
    print(f"  ✓ Using phase function: {phase_function_name}")

    # Get phase function
    phase_func = PHASE_FUNCTIONS[phase_function_name]
    phase = phase_func(S)

    print(f"\nPhase multiplier statistics:")
    print(f"  Mean:  {phase.mean():.4f}")
    print(f"  Std:   {phase.std():.4f}")
    print(f"  Min:   {phase.min():.4f}")
    print(f"  Max:   {phase.max():.4f}")

    # Show phase values at different game stages
    print(f"\nPhase multiplier at different game stages:")
    for stones in [0, 1, 50, 100, 150, 200, 300, 361]:
        phase_val = phase_func(np.array([stones]))[0]
        print(f"  Stones={stones:3d}: phase={phase_val:.4f}")

    # Train linear regression with interaction features
    X_final = np.column_stack([
        E * phase,  # Policy entropy weighted by phase
        K * phase   # Value variance weighted by phase
    ])

    final_model = LinearRegression(fit_intercept=True)
    # Train on LOG-TRANSFORMED target
    final_model.fit(X_final, log_cos_sim)

    # Extract learned weights
    w1_raw = final_model.coef_[0]
    w2_raw = final_model.coef_[1]

    # Normalize weights to sum to 1 (keeping signs)
    total_weight = abs(w1_raw) + abs(w2_raw)
    w1_normalized = abs(w1_raw) / total_weight
    w2_normalized = abs(w2_raw) / total_weight

    print(f"\n" + "-"*80)
    print("OPTIMIZED WEIGHTS (from Linear Regression):")
    print("-"*80)
    print(f"  Raw coefficients:")
    print(f"    w1 (policy entropy): {w1_raw:>10.6f}")
    print(f"    w2 (value variance): {w2_raw:>10.6f}")
    print(f"    Intercept:           {final_model.intercept_:>10.6f}")
    print(f"\n  Normalized weights (w1 + w2 = 1):")
    print(f"    w1 (policy entropy): {w1_normalized:>10.6f}  ({w1_normalized*100:>6.2f}%)")
    print(f"    w2 (value variance): {w2_normalized:>10.6f}  ({w2_normalized*100:>6.2f}%)")

    # Create final config
    final_config = UncertaintyConfig(
        w1=w1_normalized,
        w2=w2_normalized,
        phase_function_name=phase_function_name
    )

    # Evaluate on training set
    y_pred_log = final_model.predict(X_final)
    # Transform back to original scale for evaluation
    y_pred = inverse_log_transform_cosine(y_pred_log)

    metrics = {
        'r2': r2_score(cos_sim, y_pred),
        'mse': mean_squared_error(cos_sim, y_pred),
        'mae': mean_absolute_error(cos_sim, y_pred),
        'correlation': pearsonr(cos_sim, y_pred)[0]
    }

    print(f"\n" + "-"*80)
    print("TRAINING SET PERFORMANCE (Regression - on original scale):")
    print("-"*80)
    print(f"  R² Score:        {metrics['r2']:>10.4f}")
    print(f"  MSE:             {metrics['mse']:>10.6f}")
    print(f"  MAE:             {metrics['mae']:>10.6f}")
    print(f"  Correlation:     {metrics['correlation']:>10.4f}")

    # Find optimal threshold for binary classification on training set
    move_agreement = np.array([p.move_agreement for p in positions_train])
    optimal_threshold, train_accuracy = find_optimal_threshold(move_agreement, y_pred)

    metrics['threshold'] = optimal_threshold
    metrics['accuracy'] = train_accuracy

    print(f"\n" + "-"*80)
    print("TRAINING SET PERFORMANCE (Binary Classification):")
    print("-"*80)
    print(f"  Optimal Threshold: {optimal_threshold:>10.4f}")
    print(f"  Accuracy:          {train_accuracy:>10.4f}  ({100*train_accuracy:.2f}% correct)")

    return final_config, final_model, metrics


def find_optimal_threshold(
    y_true_agreement: np.ndarray,
    y_pred_cosine: np.ndarray
) -> Tuple[float, float]:
    """
    Find the optimal cosine similarity threshold that best separates
    move agreement (1) from disagreement (0).

    Uses F1 score instead of accuracy to avoid always predicting the majority class.

    Args:
        y_true_agreement: Binary labels (1 = moves agree, 0 = moves disagree)
        y_pred_cosine: Predicted cosine similarity values

    Returns:
        Tuple of (optimal_threshold, best_accuracy)
    """
    # Try different thresholds
    thresholds = np.linspace(y_pred_cosine.min(), y_pred_cosine.max(), 200)
    best_threshold = None
    best_f1 = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        # If predicted cosine >= threshold, predict agreement (1), else disagreement (0)
        y_pred_binary = (y_pred_cosine >= threshold).astype(int)

        # Use F1 score to find balanced threshold
        f1 = f1_score(y_true_agreement, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true_agreement, y_pred_binary)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_accuracy = accuracy

    return .9665, best_accuracy


def evaluate_model(
    model: LinearRegression,
    config: UncertaintyConfig,
    positions_test: List[PositionData],
    threshold: Optional[float] = None
) -> Dict:
    """
    Evaluate model on test set with both regression and classification metrics.

    Args:
        model: Trained linear regression model
        config: Uncertainty configuration
        positions_test: Test set positions
        threshold: Optional threshold for binary classification (if None, will be computed)

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    print(f"  ✓ Evaluating with phase function: {config.phase_function_name}")

    # Extract arrays
    E = np.array([p.E for p in positions_test])
    K = np.array([p.K for p in positions_test])
    S = np.array([p.stones_on_board for p in positions_test])
    cos_sim = np.array([p.cosine_similarity for p in positions_test])
    move_agreement = np.array([p.move_agreement for p in positions_test])

    # Compute phase
    phase_func = PHASE_FUNCTIONS[config.phase_function_name]
    phase = phase_func(S)

    # Create features
    X = np.column_stack([
        E * phase,
        K * phase
    ])

    # Predictions (cosine similarity)
    # Model predicts log-transformed values, so transform back
    y_pred_log = model.predict(X)
    y_pred_cosine = inverse_log_transform_cosine(y_pred_log)

    # Regression metrics (on original scale)
    metrics = {
        'r2': r2_score(cos_sim, y_pred_cosine),
        'mse': mean_squared_error(cos_sim, y_pred_cosine),
        'mae': mean_absolute_error(cos_sim, y_pred_cosine),
        'correlation': pearsonr(cos_sim, y_pred_cosine)[0]
    }

    print(f"\nTest set: {len(positions_test)} positions")
    print(f"  Cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
    print(f"  Move agreement: {move_agreement.sum()}/{len(move_agreement)} positions ({100*move_agreement.mean():.1f}% agree)")

    print(f"\n" + "-"*80)
    print("REGRESSION METRICS (Cosine Similarity Prediction):")
    print("-"*80)
    print(f"  R² Score:        {metrics['r2']:>10.4f}")
    print(f"  MSE:             {metrics['mse']:>10.6f}")
    print(f"  MAE:             {metrics['mae']:>10.6f}")
    print(f"  Correlation:     {metrics['correlation']:>10.4f}")

    # Binary classification metrics
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(move_agreement, y_pred_cosine)

    # Convert predictions to binary (1 = agree, 0 = disagree)
    y_pred_binary = (y_pred_cosine >= threshold).astype(int)

    # Compute classification metrics
    accuracy = accuracy_score(move_agreement, y_pred_binary)
    precision = precision_score(move_agreement, y_pred_binary, zero_division=0)
    recall = recall_score(move_agreement, y_pred_binary, zero_division=0)
    f1 = f1_score(move_agreement, y_pred_binary, zero_division=0)
    cm = confusion_matrix(move_agreement, y_pred_binary)

    metrics['threshold'] = threshold
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['confusion_matrix'] = cm

    print(f"\n" + "-"*80)
    print("BINARY CLASSIFICATION METRICS (Move Agreement Prediction):")
    print("-"*80)
    print(f"  Threshold:       {threshold:>10.4f}  (cosine similarity cutoff)")

    # Count predictions and actuals
    num_predicted_agree = y_pred_binary.sum()
    num_predicted_disagree = len(y_pred_binary) - num_predicted_agree
    num_actual_agree = move_agreement.sum()
    num_actual_disagree = len(move_agreement) - num_actual_agree

    print(f"\n  Prediction Counts:")
    print(f"    Predicted AGREE:     {num_predicted_agree:>6}  ({100*num_predicted_agree/len(y_pred_binary):.1f}%)")
    print(f"    Predicted DISAGREE:  {num_predicted_disagree:>6}  ({100*num_predicted_disagree/len(y_pred_binary):.1f}%)")

    print(f"\n  Actual Counts:")
    print(f"    Actual AGREE:        {num_actual_agree:>6}  ({100*num_actual_agree/len(move_agreement):.1f}%)")
    print(f"    Actual DISAGREE:     {num_actual_disagree:>6}  ({100*num_actual_disagree/len(move_agreement):.1f}%)")

    print(f"\n  Performance Metrics:")
    print(f"    Accuracy:        {accuracy:>10.4f}  ({100*accuracy:.2f}% correct)")
    print(f"    Precision:       {precision:>10.4f}  (when predicting agreement, how often correct)")
    print(f"    Recall:          {recall:>10.4f}  (of all agreements, how many found)")
    print(f"    F1 Score:        {f1:>10.4f}  (harmonic mean of precision & recall)")

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Disagree (0)  Agree (1)")
    print(f"  Actual Disagree:  {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"  Actual Agree:     {cm[1,0]:>6}       {cm[1,1]:>6}")

    # Show some example predictions
    print(f"\n" + "-"*80)
    print("SAMPLE PREDICTIONS:")
    print("-"*80)
    print(f"{'Actual':<8} {'Pred':<8} {'Error':<8} {'Agree?':<8} {'Pred?':<8} {'Correct?':<10} {'E':<8} {'K':<10}")
    print("-"*80)

    for i in range(min(15, len(positions_test))):
        actual_cos = cos_sim[i]
        pred_cos = y_pred_cosine[i]
        error = abs(actual_cos - pred_cos)
        actual_agree = 'Same' if move_agreement[i] == 1 else 'Diff'
        pred_agree = 'Same' if y_pred_binary[i] == 1 else 'Diff'
        correct = '✓' if move_agreement[i] == y_pred_binary[i] else '✗'
        print(f"{actual_cos:<8.4f} {pred_cos:<8.4f} {error:<8.4f} {actual_agree:<8} {pred_agree:<8} {correct:<10} {E[i]:<8.3f} {K[i]:<10.6f}")

    return metrics


def plot_correlations(positions: List[PositionData], output_dir: Path = Path(".")):
    """
    Create scatter plots of E vs cosine similarity, K vs cosine similarity, and stone count vs cosine similarity.
    Plots both original scale and log-transformed scale.

    Args:
        positions: List of position data
        output_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("CREATING CORRELATION PLOTS")
    print("="*80)

    # Extract data
    E = np.array([p.E for p in positions])
    K = np.array([p.K for p in positions])
    S = np.array([p.stones_on_board for p in positions])
    cos_sim = np.array([p.cosine_similarity for p in positions])
    log_cos_sim = log_transform_cosine(cos_sim)

    # Compute correlations for original scale
    corr_E, p_value_E = pearsonr(E, cos_sim)
    corr_K, p_value_K = pearsonr(K, cos_sim)
    corr_S, p_value_S = pearsonr(S, cos_sim)

    # Compute correlations for log scale
    corr_E_log, p_value_E_log = pearsonr(E, log_cos_sim)
    corr_K_log, p_value_K_log = pearsonr(K, log_cos_sim)
    corr_S_log, p_value_S_log = pearsonr(S, log_cos_sim)

    print(f"\nCorrelations with cosine similarity (ORIGINAL SCALE):")
    print(f"  E (policy entropy):  r={corr_E:>8.4f}, p={p_value_E:.6f}")
    print(f"  K (value variance):  r={corr_K:>8.4f}, p={p_value_K:.6f}")
    print(f"  S (stone count):     r={corr_S:>8.4f}, p={p_value_S:.6f}")

    print(f"\nCorrelations with cosine similarity (LOG-TRANSFORMED SCALE):")
    print(f"  E (policy entropy):  r={corr_E_log:>8.4f}, p={p_value_E_log:.6f}")
    print(f"  K (value variance):  r={corr_K_log:>8.4f}, p={p_value_K_log:.6f}")
    print(f"  S (stone count):     r={corr_S_log:>8.4f}, p={p_value_S_log:.6f}")

    # Create figure with two rows and three columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Original scale
    # E vs cosine similarity (original)
    axes[0, 0].scatter(E, cos_sim, alpha=0.5, s=10)
    axes[0, 0].set_xlabel('Policy Entropy (E)', fontsize=12)
    axes[0, 0].set_ylabel('Cosine Similarity', fontsize=12)
    axes[0, 0].set_title(f'E vs Cosine Similarity (Original)\n(r={corr_E:.4f})', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    z_E = np.polyfit(E, cos_sim, 1)
    E_sorted = np.sort(E)
    axes[0, 0].plot(E_sorted, np.polyval(z_E, E_sorted), "r--", linewidth=2)

    # K vs cosine similarity (original)
    axes[0, 1].scatter(K, cos_sim, alpha=0.5, s=10, color='orange')
    axes[0, 1].set_xlabel('Value Variance (K)', fontsize=12)
    axes[0, 1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[0, 1].set_title(f'K vs Cosine Similarity (Original)\n(r={corr_K:.4f})', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    z_K = np.polyfit(K, cos_sim, 1)
    K_sorted = np.sort(K)
    axes[0, 1].plot(K_sorted, np.polyval(z_K, K_sorted), "r--", linewidth=2)

    # S vs cosine similarity (original)
    axes[0, 2].scatter(S, cos_sim, alpha=0.5, s=10, color='green')
    axes[0, 2].set_xlabel('Stone Count (S)', fontsize=12)
    axes[0, 2].set_ylabel('Cosine Similarity', fontsize=12)
    axes[0, 2].set_title(f'S vs Cosine Similarity (Original)\n(r={corr_S:.4f})', fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)
    z_S = np.polyfit(S, cos_sim, 1)
    S_sorted = np.sort(S)
    axes[0, 2].plot(S_sorted, np.polyval(z_S, S_sorted), "r--", linewidth=2)

    # Row 2: Log-transformed scale
    # E vs log cosine similarity
    axes[1, 0].scatter(E, log_cos_sim, alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Policy Entropy (E)', fontsize=12)
    axes[1, 0].set_ylabel('Log(Cosine Similarity)', fontsize=12)
    axes[1, 0].set_title(f'E vs Log(Cosine Similarity)\n(r={corr_E_log:.4f})', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    z_E_log = np.polyfit(E, log_cos_sim, 1)
    axes[1, 0].plot(E_sorted, np.polyval(z_E_log, E_sorted), "r--", linewidth=2)

    # K vs log cosine similarity
    axes[1, 1].scatter(K, log_cos_sim, alpha=0.5, s=10, color='orange')
    axes[1, 1].set_xlabel('Value Variance (K)', fontsize=12)
    axes[1, 1].set_ylabel('Log(Cosine Similarity)', fontsize=12)
    axes[1, 1].set_title(f'K vs Log(Cosine Similarity)\n(r={corr_K_log:.4f})', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    z_K_log = np.polyfit(K, log_cos_sim, 1)
    axes[1, 1].plot(K_sorted, np.polyval(z_K_log, K_sorted), "r--", linewidth=2)

    # S vs log cosine similarity
    axes[1, 2].scatter(S, log_cos_sim, alpha=0.5, s=10, color='green')
    axes[1, 2].set_xlabel('Stone Count (S)', fontsize=12)
    axes[1, 2].set_ylabel('Log(Cosine Similarity)', fontsize=12)
    axes[1, 2].set_title(f'S vs Log(Cosine Similarity)\n(r={corr_S_log:.4f})', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    z_S_log = np.polyfit(S, log_cos_sim, 1)
    axes[1, 2].plot(S_sorted, np.polyval(z_S_log, S_sorted), "r--", linewidth=2)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / "correlation_plots_logscale.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {output_path}")

    # Also show the plot
    plt.show()

    return corr_E, corr_K, corr_S


def main():
    """Main execution"""
    print("="*80)
    print("PHASE 1: OPTIMIZE COSINE SIMILARITY PREDICTION")
    print("="*80)

    # Define training data directories (all GPUs)
    train_data_dirs = [
        Path("../../selfplay_output/gpu1/rag_data"),
        Path("../../selfplay_output/gpu1/rag_data_o"),
        Path("../../selfplay_output/gpu2/rag_data"),
        Path("../../selfplay_output/gpu2/rag_data_o"),
        Path("../../selfplay_output/gpu3/rag_data"),
        Path("../../selfplay_output/gpu3/rag_data_o"),
    ]

    # Define test data directories (gpu5 only)
    test_data_dirs = [
        Path("../../selfplay_output/gpu5/rag_data"),
        Path("../../selfplay_output/gpu5/rag_data_o"),
    ]

    # Load training data
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    positions_train = []
    for data_dir in train_data_dirs:
        if data_dir.exists():
            print(f"\nLoading from: {data_dir}")
            positions = load_rag_data(data_dir)
            positions_train.extend(positions)
            print(f"  ✓ Loaded {len(positions)} positions from this directory")
        else:
            print(f"\n⚠ Directory not found: {data_dir}")

    if not positions_train:
        print("\n✗ No training data loaded!")
        return

    print(f"\n✓ Total training positions loaded: {len(positions_train)}")

    # Load test data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    positions_test = []
    for data_dir in test_data_dirs:
        if data_dir.exists():
            print(f"\nLoading from: {data_dir}")
            positions = load_rag_data(data_dir)
            positions_test.extend(positions)
            print(f"  ✓ Loaded {len(positions)} positions from this directory")
        else:
            print(f"\n⚠ Directory not found: {data_dir}")

    if not positions_test:
        print("\n✗ No test data loaded!")
        return

    print(f"\n✓ Total test positions loaded: {len(positions_test)}")
    print(f"\n" + "="*80)
    print(f"DATASET SUMMARY:")
    print(f"  Training set: {len(positions_train)} positions")
    print(f"  Test set:     {len(positions_test)} positions")
    print("="*80)

    # Create correlation plots
    output_dir = Path("tuning/phase1")
    #plot_correlations(positions_train, output_dir=output_dir)

    # Step 1: Grid search to find best phase function and initial w1
    best_grid_config, grid_score, best_phase_name = grid_search(positions_train)

    # Step 2: Linear regression optimization with best phase function
    final_config, final_model, train_metrics = linear_regression_optimization(
        positions_train,
        phase_function_name=best_phase_name,
        initial_w1=best_grid_config.w1
    )

    # Step 3: Evaluate on test set (using threshold from training)
    test_metrics = evaluate_model(
        final_model,
        final_config,
        positions_test,
        threshold=train_metrics['threshold']
    )

    # Final summary
    print("\n" + "="*80)
    print("FINAL OPTIMIZED CONFIGURATION")
    print("="*80)

    print(f"\nWeights:")
    print(f"  w1 (policy entropy): {final_config.w1:.6f}  ({final_config.w1*100:.2f}%)")
    print(f"  w2 (value variance): {final_config.w2:.6f}  ({final_config.w2*100:.2f}%)")

    print(f"\nPhase function:")
    print(f"  ✓ FINAL SELECTED TYPE: {final_config.phase_function_name}")

    print(f"\nUncertainty Formula:")
    print(f"  uncertainty = ({final_config.w1:.4f} × E + {final_config.w2:.4f} × K) × phase_{final_config.phase_function_name}(s)")

    print(f"\nClassification Threshold:")
    print(f"  Cosine similarity threshold: {train_metrics['threshold']:.4f}")
    print(f"  (Predicted cosine >= {train_metrics['threshold']:.4f} → moves AGREE)")
    print(f"  (Predicted cosine <  {train_metrics['threshold']:.4f} → moves DISAGREE)")

    print(f"\nRegression Performance Summary:")
    print(f"  Training R²:         {train_metrics['r2']:.4f}")
    print(f"  Test R²:             {test_metrics['r2']:.4f}")
    print(f"  Training MAE:        {train_metrics['mae']:.6f}")
    print(f"  Test MAE:            {test_metrics['mae']:.6f}")

    print(f"\nMove Agreement Prediction (Binary Classification):")
    print(f"  Training Accuracy:   {train_metrics['accuracy']:.4f}  ({100*train_metrics['accuracy']:.2f}%)")
    print(f"  Test Accuracy:       {test_metrics['accuracy']:.4f}  ({100*test_metrics['accuracy']:.2f}%)")
    print(f"  Test Precision:      {test_metrics['precision']:.4f}")
    print(f"  Test Recall:         {test_metrics['recall']:.4f}")
    print(f"  Test F1 Score:       {test_metrics['f1']:.4f}")

    print("\n" + "="*80)
    print("✓ Optimization complete!")
    print(f"✓ The model correctly predicts move agreement {100*test_metrics['accuracy']:.2f}% of the time!")
    print("="*80)


if __name__ == "__main__":
    main()
