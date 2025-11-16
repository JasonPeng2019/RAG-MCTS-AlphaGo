#!/usr/bin/env python3
"""
Diagnose why sym_hashes differ between source and reconstructed data.
"""

import json
from pathlib import Path

def main():
    # Check a sample file
    rag_data_dir = Path("/scratch2/f004h1v/alphago_project/build/rag_data_1")
    sample_file = list(rag_data_dir.glob("*.json"))[0]
    
    print(f"Analyzing: {sample_file.name}")
    print("=" * 80)
    
    with open(sample_file) as f:
        data = json.load(f)
    
    print(f"\nGame settings:")
    print(f"  Rules: {data['settings']['rules']}")
    print(f"  Komi: {data['settings']['komi']}")
    print(f"  Board size: {data['settings']['board_size']}")
    
    print(f"\nFlagged positions: {len(data['flagged_positions'])}")
    
    print(f"\nAnalyzing move_number vs moves_history length discrepancy:")
    print(f"{'Pos#':<6} {'MoveNum':<10} {'HistLen':<10} {'Diff':<8} {'Player':<8} {'SymHash==StateHash':<20}")
    print("-" * 80)
    
    diffs = []
    for i, pos in enumerate(data['flagged_positions'][:15]):
        move_num = pos['move_number']
        hist_len = len(pos['moves_history'])
        diff = move_num - hist_len
        player = pos['player_to_move']
        hash_equal = pos['sym_hash'] == pos['state_hash']
        
        print(f"{i:<6} {move_num:<10} {hist_len:<10} {diff:<8} {player:<8} {hash_equal!s:<20}")
        diffs.append(diff)
    
    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        print(f"\nAverage difference: {avg_diff:.1f} moves")
        print(f"Min/Max difference: {min(diffs)}/{max(diffs)} moves")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print("""
The moves_history is consistently SHORTER than move_number by ~4-5 moves.

This is because:
1. datago_collect_search_states() is called DURING search (before move is made)
2. The search happens BEFORE the bot records the move
3. So moves_history captures moves UP TO but NOT INCLUDING the current position

This means when game_analyzer.py replays moves_history, it reconstructs a 
DIFFERENT board position than the original, leading to different sym_hashes!

The fix would be to ensure moves_history includes ALL moves up to and including
the current position, or to document that reconstruction needs to account for
this offset.
    """)

if __name__ == "__main__":
    main()
