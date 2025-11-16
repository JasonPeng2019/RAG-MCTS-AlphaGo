# Sigmoid Optimization Method - Documentation

## Overview

The sigmoid optimization method has been successfully integrated into `phase1_uncertainty_tuning.py`. This provides a third optimization approach alongside grid search and gradient optimization.

## What Was Added

### 1. New Method: `optimize_parameters_sigmoid()`

**Location**: Lines 616-764 in `phase1_uncertainty_tuning.py`

**What it does**:
- Trains a logistic regression (sigmoid) model to predict high vs. low error positions
- Uses **interaction terms**: `E*phase` and `K*phase` as features
- Optimizes phase parameters (`a`, `b`) to maximize ROC-AUC
- Learns optimal weights for policy cross-entropy and value sparseness

### 2. Updated Imports

**Location**: Lines 40-43

Added sklearn imports:
```python
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
```

### 3. Updated Method Selection

**Location**: Lines 908-913

Now supports three methods:
- `grid`: Exhaustive search (35 configurations)
- `optimize`: Gradient-based L-BFGS-B
- `sigmoid`: Logistic regression with interactions (NEW!)

### 4. Updated CLI Arguments

**Location**: Line 961

Command-line now accepts:
```bash
--method {grid,optimize,sigmoid}
```

---

## How It Works

### The Sigmoid Model

**Traditional approach (linear)**:
```
uncertainty = (w1*E + w2*K) * phase(s)
```

**Sigmoid approach (non-linear)**:
```
P(high_error) = sigmoid(w_E*(E*phase) + w_K*(K*phase) + bias)
```

Where:
- `E` = policy cross-entropy
- `K` = value sparseness
- `phase(s)` = a*(stones/361) + b
- `sigmoid(z)` = 1 / (1 + exp(-z))

### Optimization Process

1. **Create Binary Labels**: Split errors at median into high/low classes
2. **Define Objective**: Maximize ROC-AUC (separation of high/low error)
3. **Optimize Phase**: Use L-BFGS-B to find optimal `a` and `b`
4. **Train Logistic Regression**: Learn weights `w_E`, `w_K`, `bias`
5. **Extract Normalized Weights**: Convert to `w1`, `w2` format

### Key Features

**Interaction Terms**:
```python
X_interactions = np.column_stack([
    E * phase,  # How policy uncertainty varies with game phase
    K * phase   # How value uncertainty varies with game phase
])
```

This captures how uncertainty features interact with game phase, unlike the linear model which only multiplies.

**Classification-Based**:
- Treats uncertainty detection as binary classification
- Optimizes for separating high-error from low-error positions
- Uses ROC-AUC metric (area under receiver operating characteristic curve)

---

## Usage

### Basic Usage

```bash
python phase1_uncertainty_tuning.py \
    --shallow-db ./data/shallow_mcts_db.json \
    --deep-db ./data/deep_mcts_db.json \
    --method sigmoid \
    --output-dir ./tuning_results/phase1_sigmoid
```

### Example Output

```
================================================================================
SIGMOID OPTIMIZATION: Training sigmoid model with interactions
================================================================================
Training on 8000 positions
  E (policy cross-entropy) range: [1.234, 5.678]
  K (value sparseness) range: [0.001, 0.089]
  Errors range: [0.000023, 0.234567]

Binary classification:
  Error threshold: 0.012345
  High-error positions: 4000 (50.0%)
  Low-error positions: 4000 (50.0%)

Optimizing phase function parameters...
[L-BFGS-B iterations...]

Optimal phase function: 0.2347*s + 0.8623
ROC-AUC achieved: 0.8456

Learned sigmoid model:
  uncertainty = sigmoid(2.4567*E*phase + 1.2345*K*phase + -0.5432)

================================================================================
SIGMOID OPTIMIZATION COMPLETE:
  Normalized w1 (policy): 0.6654
  Normalized w2 (value): 0.3346
  Phase function: 0.2347*s + 0.8623
  ROC-AUC: 0.8456
================================================================================
```

---

## Comparison of Methods

| Method | Speed | Optimization Metric | Output Type | Best For |
|--------|-------|-------------------|-------------|----------|
| **Grid** | Slow (30 min) | Pearson correlation | Linear weights | Thorough exploration |
| **Optimize** | Fast (5 min) | Pearson correlation | Linear weights | Quick results |
| **Sigmoid** | Medium (10 min) | ROC-AUC | Probability scores | Classification tasks |

### When to Use Sigmoid

**Use sigmoid if**:
- You want probabilistic uncertainty scores (0 to 1)
- You care about separating high/low error positions (classification)
- You want to model interactions between features and game phase
- You prefer non-linear decision boundaries

**Use optimize/grid if**:
- You want simple linear combinations
- You want direct correlation with continuous error values
- You prefer interpretability over performance

---

## Technical Details

### Interaction Terms Explained

**Without interactions (linear model)**:
```python
uncertainty = (w1*E + w2*K) * phase
```
- Phase multiplies the entire weighted sum
- E and K don't "interact" with phase differently

**With interactions (sigmoid model)**:
```python
features = [E*phase, K*phase]
uncertainty = sigmoid(w_E*(E*phase) + w_K*(K*phase) + bias)
```
- E and K can have different relationships with phase
- Model learns how much each feature's importance changes with game phase
- More flexible: can capture "E matters more in opening, K matters more in endgame"

### ROC-AUC Metric

**ROC-AUC** (Receiver Operating Characteristic - Area Under Curve):
- Measures how well the model separates high-error from low-error positions
- Range: 0.5 (random) to 1.0 (perfect separation)
- Good values: 0.7-0.8 (acceptable), 0.8-0.9 (good), 0.9+ (excellent)

### Fallback Behavior

If sklearn is not available:
```python
if not HAS_SKLEARN:
    print("ERROR: sklearn is required for sigmoid optimization")
    print("Falling back to standard optimization...")
    return self.optimize_parameters()
```

The method automatically falls back to gradient optimization if sklearn is missing.

---

## Output Files

Same as other methods:
- `best_uncertainty_config.json` - Optimal configuration
- `threshold_analysis.json` - Storage threshold recommendations
- `threshold_pareto_frontier.png` - Visualization

Additional information in JSON:
```json
{
  "config": {
    "w1": 0.6654,
    "w2": 0.3346,
    "phase_function_type": "linear",
    "phase_coefficients": [0.2347, 0.8623]
  },
  "validation_metrics": {
    "pearson_correlation": 0.7823,
    "roc_auc": 0.8456
  },
  "tuning_method": "sigmoid"
}
```

---

## Dependencies

Required packages:
```bash
pip install scikit-learn  # For LogisticRegression and roc_auc_score
```

Already required:
- numpy
- scipy
- matplotlib

---

## Limitations

1. **Only linear phase functions**: Currently only supports `phase(s) = a*s + b`
2. **Binary classification**: Converts continuous errors to binary labels (loses some information)
3. **Requires balanced data**: Works best when ~50% high-error and ~50% low-error positions
4. **sklearn dependency**: Won't work without scikit-learn installed

---

## Future Enhancements

Potential improvements:
1. Support for exponential/piecewise phase functions in sigmoid mode
2. Multi-class classification (low/medium/high error)
3. Weighted logistic regression (weight by error magnitude)
4. Cross-validation for phase parameter selection
5. Feature engineering (polynomial interactions, etc.)

---

## Questions?

For issues or questions about the sigmoid method, refer to the main `phase1_uncertainty_tuning.py` code or check the parameter tuning plan document.
