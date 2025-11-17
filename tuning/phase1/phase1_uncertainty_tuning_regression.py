#!/usr/bin/env python3
"""
Phase 1 Uncertainty Tuning - Regression Data Extraction

This script extracts data from selfplay RAG JSON files for regression analysis.
It collects:
- E (policy_entropy) and K (value_variance) from uncertainty_metrics
- Deep search: value and prior for the best move
- Shallow search: value and prior for the best move
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PositionData:
    """Data extracted from a flagged position"""
    # Identifiers
    game_id: str
    query_id: str
    move_number: int
    stones_on_board: int

    # Uncertainty metrics
    E: float  # policy_entropy
    K: float  # value_variance

    # Shallow search (root level)
    shallow_best_move: str
    shallow_value: Optional[float]
    shallow_prior: Optional[float]
    shallow_visits: Optional[int]

    # Deep search (deep_result)
    deep_best_move: str
    deep_value: Optional[float]
    deep_prior: Optional[float]
    deep_visits: Optional[int]
    deep_total_visits: int

    # Move agreement
    moves_agree: bool


def find_child_stats(children: List[Dict], move: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Find the value, prior, and visits for a specific move in the children list.

    Args:
        children: List of child move dictionaries
        move: Move string (e.g., "G4")

    Returns:
        Tuple of (value, prior, visits) or (None, None, None) if not found
    """
    for child in children:
        if child.get("move") == move:
            return (
                child.get("value"),
                child.get("prior"),
                child.get("visits")
            )
    return None, None, None


def extract_position_data(flagged_pos: Dict, game_id: str) -> Optional[PositionData]:
    """
    Extract all relevant data from a single flagged position.

    Args:
        flagged_pos: Dictionary for one flagged position
        game_id: Game identifier string

    Returns:
        PositionData object or None if data is incomplete
    """
    try:
        # Extract basic info
        query_id = flagged_pos.get("query_id", "")
        move_number = flagged_pos.get("move_number", 0)
        stones_on_board = flagged_pos.get("stone_count", {}).get("total", 0)

        # Extract uncertainty metrics
        uncertainty = flagged_pos.get("uncertainty_metrics", {})
        E = uncertainty.get("policy_entropy")
        K = uncertainty.get("value_variance")

        if E is None or K is None:
            print(f"  ⚠ Missing uncertainty metrics for {query_id}")
            return None

        # Extract shallow search best move
        shallow_best_move_obj = flagged_pos.get("best_move", {})
        shallow_best_move = shallow_best_move_obj.get("move")

        if not shallow_best_move:
            print(f"  ⚠ Missing shallow best_move for {query_id}")
            return None

        # Find shallow move stats in children
        shallow_children = flagged_pos.get("children", [])
        shallow_value, shallow_prior, shallow_visits = find_child_stats(
            shallow_children, shallow_best_move
        )

        # Extract deep search data
        deep_result = flagged_pos.get("deep_result", {})

        if not deep_result.get("available") or deep_result.get("status") != "ok":
            print(f"  ⚠ Deep result not available for {query_id}")
            return None

        deep_best_move_obj = deep_result.get("best_move", {})
        deep_best_move = deep_best_move_obj.get("move")
        deep_total_visits = deep_result.get("visits", 0)

        if not deep_best_move:
            print(f"  ⚠ Missing deep best_move for {query_id}")
            return None

        # Find deep move stats in deep_result children
        deep_children = deep_result.get("children", [])
        deep_value, deep_prior, deep_visits = find_child_stats(
            deep_children, deep_best_move
        )

        # Check if moves agree
        moves_agree = (shallow_best_move == deep_best_move)

        return PositionData(
            game_id=game_id,
            query_id=query_id,
            move_number=move_number,
            stones_on_board=stones_on_board,
            E=E,
            K=K,
            shallow_best_move=shallow_best_move,
            shallow_value=shallow_value,
            shallow_prior=shallow_prior,
            shallow_visits=shallow_visits,
            deep_best_move=deep_best_move,
            deep_value=deep_value,
            deep_prior=deep_prior,
            deep_visits=deep_visits,
            deep_total_visits=deep_total_visits,
            moves_agree=moves_agree
        )

    except Exception as e:
        print(f"  ✗ Error processing position: {e}")
        return None


def load_rag_data(rag_data_dir: Path) -> List[PositionData]:
    """
    Load all RAG data from JSON files in the directory.

    Args:
        rag_data_dir: Path to directory containing RAG JSON files

    Returns:
        List of PositionData objects
    """
    all_positions = []

    # Find all JSON files
    json_files = list(rag_data_dir.glob("RAG_rawdata_game_*.json"))

    if not json_files:
        print(f"⚠ No JSON files found in {rag_data_dir}")
        return []

    print(f"\nFound {len(json_files)} JSON files")
    print("="*80)

    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            game_id = data.get("game_id", "")
            flagged_positions = data.get("flagged_positions", [])

            print(f"  Game ID: {game_id}")
            print(f"  Flagged positions: {len(flagged_positions)}")

            for i, flagged_pos in enumerate(flagged_positions):
                pos_data = extract_position_data(flagged_pos, game_id)
                if pos_data:
                    all_positions.append(pos_data)

            print(f"  ✓ Extracted {len([p for p in all_positions if p.game_id == game_id])} positions")

        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
            continue

    return all_positions


def print_statistics(positions: List[PositionData]):
    """Print statistics about the extracted data"""
    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)

    print(f"\nTotal positions extracted: {len(positions)}")

    if not positions:
        return

    # Extract arrays
    E_values = np.array([p.E for p in positions])
    K_values = np.array([p.K for p in positions])
    stones = np.array([p.stones_on_board for p in positions])

    # Move agreement
    agreements = sum(1 for p in positions if p.moves_agree)
    disagreements = len(positions) - agreements
    agreement_rate = agreements / len(positions) * 100

    print(f"\nMove Agreement:")
    print(f"  Shallow = Deep: {agreements} ({agreement_rate:.1f}%)")
    print(f"  Shallow ≠ Deep: {disagreements} ({100-agreement_rate:.1f}%)")

    print(f"\nPolicy Entropy (E):")
    print(f"  Mean: {np.mean(E_values):.6f}")
    print(f"  Std:  {np.std(E_values):.6f}")
    print(f"  Min:  {np.min(E_values):.6f}")
    print(f"  Max:  {np.max(E_values):.6f}")

    print(f"\nValue Variance (K):")
    print(f"  Mean: {np.mean(K_values):.6f}")
    print(f"  Std:  {np.std(K_values):.6f}")
    print(f"  Min:  {np.min(K_values):.6f}")
    print(f"  Max:  {np.max(K_values):.6f}")

    print(f"\nStones on Board:")
    print(f"  Mean: {np.mean(stones):.1f}")
    print(f"  Min:  {np.min(stones)}")
    print(f"  Max:  {np.max(stones)}")

    # Check for missing values
    shallow_value_missing = sum(1 for p in positions if p.shallow_value is None)
    deep_value_missing = sum(1 for p in positions if p.deep_value is None)

    print(f"\nMissing Values:")
    print(f"  Shallow value missing: {shallow_value_missing}")
    print(f"  Deep value missing:    {deep_value_missing}")


def print_sample_positions(positions: List[PositionData], n: int = 5):
    """Print sample positions with all extracted values"""
    print("\n" + "="*80)
    print(f"SAMPLE POSITIONS (first {n})")
    print("="*80)

    for i, pos in enumerate(positions[:n]):
        print(f"\nPosition {i+1}:")
        print(f"  Query ID: {pos.query_id}")
        print(f"  Move number: {pos.move_number}")
        print(f"  Stones on board: {pos.stones_on_board}")
        print(f"  ")
        print(f"  Uncertainty Metrics:")
        print(f"    E (policy_entropy):  {pos.E:.6f}")
        print(f"    K (value_variance):  {pos.K:.6f}")
        print(f"  ")
        print(f"  Shallow Search:")
        print(f"    Best move: {pos.shallow_best_move}")
        shallow_val_str = f"{pos.shallow_value:.6f}" if pos.shallow_value is not None else "N/A"
        shallow_prior_str = f"{pos.shallow_prior:.6f}" if pos.shallow_prior is not None else "N/A"
        shallow_visits_str = str(pos.shallow_visits) if pos.shallow_visits is not None else "N/A"
        print(f"    Value:     {shallow_val_str}")
        print(f"    Prior:     {shallow_prior_str}")
        print(f"    Visits:    {shallow_visits_str}")
        print(f"  ")
        print(f"  Deep Search ({pos.deep_total_visits} total visits):")
        print(f"    Best move: {pos.deep_best_move}")
        deep_val_str = f"{pos.deep_value:.6f}" if pos.deep_value is not None else "N/A"
        deep_prior_str = f"{pos.deep_prior:.6f}" if pos.deep_prior is not None else "N/A"
        deep_visits_str = str(pos.deep_visits) if pos.deep_visits is not None else "N/A"
        print(f"    Value:     {deep_val_str}")
        print(f"    Prior:     {deep_prior_str}")
        print(f"    Visits:    {deep_visits_str}")
        print(f"  ")
        print(f"  Agreement: {'✓ YES' if pos.moves_agree else '✗ NO'}")
        print("-" * 80)


def main():
    """Main execution function"""
    # Path to RAG data directory
    rag_data_dir = Path("/scratch2/f004h1v/alphago_project/selfplay_output/gpu1/rag_data")

    if not rag_data_dir.exists():
        print(f"✗ Directory not found: {rag_data_dir}")
        return

    print("="*80)
    print("PHASE 1 UNCERTAINTY TUNING - DATA EXTRACTION")
    print("="*80)
    print(f"\nRAG data directory: {rag_data_dir}")

    # Load all data
    positions = load_rag_data(rag_data_dir)

    if not positions:
        print("\n✗ No data extracted!")
        return

    # Print statistics
    print_statistics(positions)

    # Print sample positions
    print_sample_positions(positions, n=5)

    print("\n" + "="*80)
    print("✓ Data extraction complete!")
    print("="*80)


if __name__ == "__main__":
    main()
