#!/usr/bin/env python3
"""
Test if we can reconstruct positions by using the COMPLETE game history.
We'll extract a full game's moves and replay up to each flagged position.
"""

import json
from pathlib import Path

def main():
    # Pick a game file
    game_file = Path("/scratch2/f004h1v/alphago_project/build/rag_data_1/RAG_rawdata_game_00580397BB42EC5DDA8F81ED53A32D13.json")
    
    print("="*80)
    print("TESTING POSITION RECONSTRUCTION")
    print("="*80)
    
    with open(game_file) as f:
        data = json.load(f)
    
    game_id = data.get('game_id', 'unknown')
    settings = data.get('settings', {})
    flagged_positions = data.get('flagged_positions', [])
    
    print(f"\nGame: {game_id}")
    print(f"Settings: komi={settings.get('komi')}, board_size={settings.get('board_size')}, rules={settings.get('rules')}")
    print(f"Flagged positions: {len(flagged_positions)}")
    
    # Look at the first few positions
    print(f"\n{'='*80}")
    print("ANALYZING FLAGGED POSITIONS")
    print(f"{'='*80}")
    
    for i, pos in enumerate(flagged_positions[:5]):
        move_num = pos.get('move_number')
        hist_len = len(pos.get('moves_history', []))
        gap = move_num - hist_len
        
        print(f"\nPosition {i}:")
        print(f"  move_number: {move_num}")
        print(f"  moves_history length: {hist_len}")
        print(f"  gap: {gap}")
        print(f"  source sym_hash: {pos.get('sym_hash')}")
        
        if hist_len > 0:
            print(f"  first move in history: {pos['moves_history'][0]}")
            print(f"  last move in history: {pos['moves_history'][-1]}")
    
    # Now check: can we reconstruct the full game by accumulating moves?
    print(f"\n{'='*80}")
    print("ATTEMPTING TO RECONSTRUCT FULL GAME SEQUENCE")
    print(f"{'='*80}")
    
    # The key insight: moves_history in later positions should contain earlier moves
    # Let's see if we can build the full sequence by looking at the longest moves_history
    
    longest_history = []
    longest_idx = -1
    
    for i, pos in enumerate(flagged_positions):
        hist = pos.get('moves_history', [])
        if len(hist) > len(longest_history):
            longest_history = hist
            longest_idx = i
    
    print(f"\nLongest moves_history found:")
    print(f"  Position index: {longest_idx}")
    print(f"  Length: {len(longest_history)}")
    print(f"  move_number at that position: {flagged_positions[longest_idx].get('move_number')}")
    
    # Check if this is the complete game
    summary = data.get('summary', {})
    total_moves = summary.get('total_moves', 0)
    
    print(f"\nGame summary:")
    print(f"  total_moves: {total_moves}")
    print(f"  Longest history covers: {len(longest_history)} moves")
    
    if len(longest_history) == total_moves:
        print(f"\n✓ The longest moves_history contains the COMPLETE game!")
        print(f"  This means we CAN reconstruct positions if we use the full game sequence.")
    else:
        print(f"\n✗ Even the longest moves_history is incomplete.")
        print(f"  Missing {total_moves - len(longest_history)} moves.")
    
    # Now test: for position at move_number N, can we use first N moves from longest_history?
    print(f"\n{'='*80}")
    print("TESTING RECONSTRUCTION STRATEGY")
    print(f"{'='*80}")
    
    print(f"\nStrategy: Use the complete game moves (from longest history),")
    print(f"and replay up to move_number for each flagged position.")
    print(f"\nExample for first position:")
    
    first_pos = flagged_positions[0]
    move_num = first_pos.get('move_number')
    
    print(f"  Original position:")
    print(f"    move_number: {move_num}")
    print(f"    stored moves_history: {len(first_pos.get('moves_history', []))} moves")
    print(f"    source sym_hash: {first_pos.get('sym_hash')}")
    
    print(f"\n  Reconstructed approach:")
    print(f"    Use first {move_num} moves from complete game")
    if move_num <= len(longest_history):
        reconstructed_moves = longest_history[:move_num]
        print(f"    Would replay: {len(reconstructed_moves)} moves")
        print(f"    First few: {reconstructed_moves[:3]}")
    else:
        print(f"    ✗ Cannot reconstruct - move_number ({move_num}) > longest_history ({len(longest_history)})")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if len(longest_history) >= total_moves:
        print("\n✓ Reconstruction IS POSSIBLE!")
        print("  The complete game sequence exists in the data.")
        print("  To reconstruct any flagged position:")
        print("    1. Find the longest moves_history in the game")
        print("    2. Take the first N moves (where N = move_number)")
        print("    3. Replay those moves with KataGo")
        print("    4. The sym_hash should match the original")
    else:
        print("\n✗ Reconstruction NOT POSSIBLE from this data alone.")
        print(f"  The game had {total_moves} moves, but longest history has {len(longest_history)}.")
        print("  The missing moves cannot be recovered from the stored data.")

if __name__ == "__main__":
    main()
