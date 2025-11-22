#!/usr/bin/env python3
"""
test_rag_retrieval.py

Utility script to inspect and score cached RAG entries for a given sym_hash.
Prints the scoring breakdown (phase similarity, policy cosine, komi/score deltas)
so you can verify retrieval behaviour without running a full match.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from run_datago_recursive_match import RecursiveDataGoPlayer


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def describe_contexts(ctx, current_phase: int, current_policy: np.ndarray,
                      current_komi: float, current_score: float,
                      visits_normalizer: float):
    """Print scoring metrics for each stored context."""
    phase_norm = current_phase / 361.0
    print(f"Evaluating {len(ctx.contexts)} cached contexts...")
    for idx, entry in enumerate(ctx.contexts):
        ctx_phase_norm = entry['game_phase'] / 361.0
        phase_cos = 1.0 - abs(ctx_phase_norm - phase_norm)
        policy_cos = cosine(entry['policy'], current_policy)
        ctx_meta = entry['metadata'] or {}
        raw_entry = ctx_meta.get('raw_entry') or {}
        ctx_komi = raw_entry.get('komi', ctx_meta.get('komi', current_komi))
        komi_delta = ctx_komi - current_komi
        score_delta = entry['score_lead'] - current_score
        visits_score = min(entry['deep_visits'] / max(visits_normalizer, 1.0), 1.0)
        total = (
            0.5 * phase_cos +
            0.4 * policy_cos +
            0.3 * visits_score -
            0.05 * abs(komi_delta) -
            0.05 * abs(score_delta)
        )
        print(
            f"[{idx}] move={entry['move']:>5} "
            f"phase_cos={phase_cos:>6.3f} policy_cos={policy_cos:>6.3f} visits={visits_score:>6.3f} "
            f"komi_delta={komi_delta:+.2f} score_delta={score_delta:+.2f} "
            f"deep_visits={entry['deep_visits']:>5} total={total:>6.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect RAG retrieval for a sym_hash")
    parser.add_argument("--config", required=True, help="Path to datago config.yaml")
    parser.add_argument("--sym-hash", required=True, help="Symmetry hash to inspect")
    parser.add_argument("--phase", type=int, default=150, help="Current stones on board")
    parser.add_argument("--komi", type=float, default=7.5, help="Current komi")
    parser.add_argument("--score-lead", type=float, default=0.0, help="Current score lead")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Instantiate player (no KataGo controller needed for cache lookups)
    player = RecursiveDataGoPlayer(config, katago=None)

    context = player.query_rag_exact(args.sym_hash)
    if not context:
        print(f"No cached entry found for sym_hash {args.sym_hash}")
        return

    # Use the first stored policy as the reference unless user overrides
    current_policy: Optional[np.ndarray] = None
    if context.contexts:
        current_policy = context.contexts[0]['policy']
    else:
        current_policy = np.zeros(player.board_state.board_size ** 2, dtype=np.float32)

    describe_contexts(
        context,
        current_phase=args.phase,
        current_policy=current_policy,
        current_komi=args.komi,
        current_score=args.score_lead,
        visits_normalizer=player.visits_normalizer,
    )

    best, score = context.get_best_context(
        current_phase=args.phase,
        current_policy=current_policy,
        current_komi=args.komi,
        current_score_lead=args.score_lead,
        visits_normalizer=player.visits_normalizer,
    )
    if best:
        print("\nSelected context:")
        print(f"  move:        {best['move']}")
        print(f"  deep_visits: {best['deep_visits']}")
        print(f"  winrate:     {best['winrate']:.3f}")
        print(f"  score_lead:  {best['score_lead']:.3f}")
    else:
        print("No suitable context selected.")


if __name__ == "__main__":
    main()
