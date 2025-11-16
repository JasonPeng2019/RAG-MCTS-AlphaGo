#!/usr/bin/env python3
"""
Test reconstruction of a single position using the game_analyzer infrastructure.
Extract a position from rag_data, get its complete move sequence,
reconstruct it, and compare sym_hashes.
"""

import json
import sys
import os
from pathlib import Path

# Use GPU 1 (which is free)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add the parent directory to path to import game_analyzer
sys.path.insert(0, str(Path(__file__).parent))

from game_analyzer import KataGoAnalyzer

def main():
    # Load a test game
    game_file = Path("/scratch2/f004h1v/alphago_project/build/rag_data_1/RAG_rawdata_game_00580397BB42EC5DDA8F81ED53A32D13.json")
    
    with open(game_file) as f:
        data = json.load(f)
    
    game_id = data.get('game_id')
    settings = data.get('settings', {})
    flagged_positions = data.get('flagged_positions', [])
    
    # Find the complete game sequence (longest moves_history)
    complete_game = []
    for pos in flagged_positions:
        hist = pos.get('moves_history', [])
        if len(hist) > len(complete_game):
            complete_game = hist
    
    print("="*80)
    print("SINGLE POSITION RECONSTRUCTION TEST")
    print("="*80)
    print(f"\nGame: {game_id}")
    print(f"Board size: {settings.get('board_size')}")
    print(f"Komi: {settings.get('komi')}")
    print(f"Rules: {settings.get('rules')}")
    print(f"Complete game: {len(complete_game)} moves")
    print(f"Flagged positions: {len(flagged_positions)}")
    
    # Test position 0 (move 3)
    test_pos = flagged_positions[0]
    move_num = test_pos.get('move_number')
    original_hash = test_pos.get('sym_hash')
    
    print(f"\n{'='*80}")
    print("TEST POSITION")
    print(f"{'='*80}")
    print(f"Position index: 0")
    print(f"move_number: {move_num}")
    print(f"Original sym_hash: {original_hash}")
    print(f"Stored moves_history length: {len(test_pos.get('moves_history', []))}")
    
    # Get the complete move sequence up to this position
    reconstruct_moves = complete_game[:move_num]
    print(f"\nReconstructing with first {len(reconstruct_moves)} moves from complete game:")
    print(f"  {reconstruct_moves}")
    
    # Initialize KataGo analyzer
    print(f"\n{'='*80}")
    print("INITIALIZING KATAGO")
    print(f"{'='*80}")
    
    import time
    import select
    analyzer = KataGoAnalyzer(
        katago_path="/scratch2/f004h1v/alphago_project/build/katago",
        model_path="/scratch2/f004h1v/alphago_project/katago_repo/run/kata1-b28c512nbt-s11653980416-d5514111622.bin.gz",
        config_path="/scratch2/f004h1v/alphago_project/katago_repo/run/analysis.cfg"
    )
    
    # Give KataGo time to fully initialize and check stderr
    print("Waiting for KataGo to initialize neural network...")
    time.sleep(10)
    
    # Check if process is alive
    if analyzer.katago.poll() is not None:
        print(f"ERROR: KataGo process died with code {analyzer.katago.returncode}")
        print("STDERR:")
        stderr = analyzer.katago.stderr.read()
        print(stderr)
        return
    
    print("Ready.")
    
    # Reconstruct the position
    print(f"\n{'='*80}")
    print("RECONSTRUCTING POSITION")
    print(f"{'='*80}")
    
    try:
        result = analyzer.analyze_position(
            moves=reconstruct_moves,
            komi=settings.get('komi'),
            board_size=settings.get('board_size'),
            rules=settings.get('rules')
        )
        
        if result and hasattr(result, 'sym_hash'):
            reconstructed_hash = result.sym_hash
            
            print(f"\nOriginal sym_hash:      {original_hash}")
            print(f"Reconstructed sym_hash: {reconstructed_hash}")
            
            if reconstructed_hash == original_hash:
                print("\n" + "="*80)
                print("✓✓✓ SUCCESS! HASHES MATCH! ✓✓✓")
                print("="*80)
                print("\nThis proves that reconstruction WORKS when using the")
                print("complete game sequence instead of the incomplete moves_history!")
            else:
                print("\n" + "="*80)
                print("✗ HASHES DO NOT MATCH")
                print("="*80)
                print("\nPossible reasons:")
                print("1. Rules/komi don't match exactly")
                print("2. Move sequence is still incorrect")
                print("3. Board symmetry handling differs")
        else:
            print("\n✗ Failed to get sym_hash from reconstructed position")
            print(f"Result: {result}")
            
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()
