#!/usr/bin/env python3
"""
Plot E (policy entropy) vs scaled cosine similarity

This script loads the training data and creates a plot showing the relationship
between policy entropy and the log-scaled cosine similarity.
"""

import json
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass
from scipy.stats import pearsonr
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


def log_transform_cosine(cos_sim: np.ndarray) -> np.ndarray:
    """
    Transform cosine similarity to spread out values near 1 and compress values far from 1.

    Uses -log(1 - x) transformation which:
    - Spreads out values very close to 1 (e.g., 0.99, 0.999)
    - Compresses values farther from 1 (e.g., 0, -1)

    Args:
        cos_sim: Cosine similarity values in range [-1, 1]

    Returns:
        Log-transformed values (unbounded positive values)
    """
    # Use -log(1 - cos_sim) to spread out values near 1
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    return -np.log(1.0 - cos_sim + epsilon)


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


def plot_E_vs_scaled_cosine(positions: List[PositionData], output_dir: Path = Path(".")):
    """
    Create a plot of E (policy entropy) vs scaled cosine similarity.

    Args:
        positions: List of position data
        output_dir: Directory to save plot
    """
    print("\n" + "="*80)
    print("CREATING E vs SCALED COSINE SIMILARITY PLOT")
    print("="*80)

    # Extract data
    E = np.array([p.E for p in positions])
    cos_sim = np.array([p.cosine_similarity for p in positions])
    scaled_cos_sim = log_transform_cosine(cos_sim)

    # Compute correlation
    corr, p_value = pearsonr(E, scaled_cos_sim)

    print(f"\nData statistics:")
    print(f"  Number of positions: {len(positions)}")
    print(f"  E range: [{E.min():.3f}, {E.max():.3f}]")
    print(f"  E mean: {E.mean():.3f}")
    print(f"  Cosine similarity range: [{cos_sim.min():.4f}, {cos_sim.max():.4f}]")
    print(f"  Cosine similarity mean: {cos_sim.mean():.4f}")
    print(f"  Scaled cosine range: [{scaled_cos_sim.min():.4f}, {scaled_cos_sim.max():.4f}]")
    print(f"  Scaled cosine mean: {scaled_cos_sim.mean():.4f}")

    print(f"\nCorrelation:")
    print(f"  E vs Scaled Cosine Similarity: r={corr:.4f}, p={p_value:.6f}")

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    ax.scatter(E, scaled_cos_sim, alpha=0.5, s=20, color='blue')
    ax.set_xlabel('Policy Entropy (E)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Scaled Cosine Similarity\n[-log(1 - cos_sim)]', fontsize=14, fontweight='bold')
    ax.set_title(f'Policy Entropy vs Scaled Cosine Similarity\n(r={corr:.4f}, p={p_value:.6f})',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(E, scaled_cos_sim, 1)
    E_sorted = np.sort(E)
    ax.plot(E_sorted, np.polyval(z, E_sorted), "r--", linewidth=2.5,
            label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.4f}')
    ax.legend(fontsize=12, loc='best')

    # Add text box with transformation info
    textstr = 'Transformation: -log(1 - cos_sim)\nSpreads values near 1, compresses far from 1'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / "E_vs_scaled_cosine_similarity.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    # Show the plot
    plt.show()

    return corr, p_value


def main():
    """Main execution"""
    print("="*80)
    print("PLOT E vs SCALED COSINE SIMILARITY")
    print("="*80)

    # Define training data directories
    train_data_dirs = [
        Path("../../selfplay_output/gpu1/rag_data"),
        Path("../../selfplay_output/gpu1/rag_data_o"),
        Path("../../selfplay_output/gpu2/rag_data"),
        Path("../../selfplay_output/gpu2/rag_data_o"),
        Path("../../selfplay_output/gpu3/rag_data"),
        Path("../../selfplay_output/gpu3/rag_data_o"),
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

    # Balance the training set by undersampling the agree class
    print("\n" + "="*80)
    print("BALANCING TRAINING SET")
    print("="*80)

    # Separate agree and disagree cases
    positions_agree = [p for p in positions_train if p.move_agreement == 1]
    positions_disagree = [p for p in positions_train if p.move_agreement == 0]

    print(f"\nBefore balancing:")
    print(f"  Agree cases:    {len(positions_agree)} ({100*len(positions_agree)/len(positions_train):.1f}%)")
    print(f"  Disagree cases: {len(positions_disagree)} ({100*len(positions_disagree)/len(positions_train):.1f}%)")
    print(f"  Total:          {len(positions_train)}")

    # Undersample the majority class (agree) to match minority class (disagree)
    num_disagree = len(positions_disagree)

    if len(positions_agree) > num_disagree:
        # Randomly sample from agree cases
        import random
        random.seed(42)  # For reproducibility
        positions_agree_sampled = random.sample(positions_agree, num_disagree)

        # Combine balanced dataset
        positions_train = positions_agree_sampled + positions_disagree

        # Shuffle the combined list
        random.shuffle(positions_train)

        print(f"\nAfter balancing (undersampling agree cases):")
        print(f"  Agree cases:    {len(positions_agree_sampled)} ({100*len(positions_agree_sampled)/len(positions_train):.1f}%)")
        print(f"  Disagree cases: {len(positions_disagree)} ({100*len(positions_disagree)/len(positions_train):.1f}%)")
        print(f"  Total:          {len(positions_train)}")
    else:
        print(f"\n⚠ No balancing needed - disagree cases are already more common or equal")

    # Create plot
    output_dir = Path("tuning/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_E_vs_scaled_cosine(positions_train, output_dir=output_dir)

    print("\n" + "="*80)
    print("✓ Plot generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
