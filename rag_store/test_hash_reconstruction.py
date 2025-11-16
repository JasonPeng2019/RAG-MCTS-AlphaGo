#!/usr/bin/env python3
"""
Actually reconstruct a position by replaying moves through KataGo
and compare the resulting sym_hash with the original.
"""

import json
import subprocess
import time
from pathlib import Path

class KataGoAnalyzer:
    """Persistent KataGo process for analysis."""
    
    def __init__(self):
        katago_path = "/scratch2/f004h1v/alphago_project/build/katago"
        model_path = "/scratch2/f004h1v/alphago_project/build/models/model.bin.gz"
        config_path = "/scratch2/f004h1v/alphago_project/katago_repo/run/analysis.cfg"
        
        cmd = [katago_path, "analysis", "-model", model_path, "-config", config_path]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait for KataGo to initialize
        print("Initializing KataGo engine...")
        time.sleep(5)
        print("KataGo ready.")
    
    def query(self, moves, komi, board_size, rules):
        """Send a query and get sym_hash."""
        query = {
            "id": "test",
            "moves": moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "includePolicy": False,
            "includeOwnership": False,
            "analyzeTurns": [len(moves)]
        }
        
        # Send query
        query_str = json.dumps(query) + "\n"
        self.process.stdin.write(query_str)
        self.process.stdin.flush()
        
        # Read response
        response_line = self.process.stdout.readline()
        if response_line:
            try:
                response = json.loads(response_line)
                if 'rootInfo' in response and 'symHash' in response['rootInfo']:
                    return response['rootInfo']['symHash']
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response: {response_line[:200]}")
        
        return None
    
    def close(self):
        """Shutdown KataGo."""
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait(timeout=5)

def main():
    # Load the test game
    game_file = Path("/scratch2/f004h1v/alphago_project/build/rag_data_1/RAG_rawdata_game_00580397BB42EC5DDA8F81ED53A32D13.json")
    
    with open(game_file) as f:
        data = json.load(f)
    
    game_id = data.get('game_id')
    settings = data.get('settings', {})
    flagged_positions = data.get('flagged_positions', [])
    
    # Find the longest moves_history (complete game)
    complete_game = []
    for pos in flagged_positions:
        hist = pos.get('moves_history', [])
        if len(hist) > len(complete_game):
            complete_game = hist
    
    print("="*80)
    print("TESTING POSITION RECONSTRUCTION WITH KATAGO")
    print("="*80)
    print(f"\nGame: {game_id}")
    print(f"Settings: komi={settings.get('komi')}, board_size={settings.get('board_size')}")
    print(f"Complete game sequence: {len(complete_game)} moves")
    
    # Test the first 3 flagged positions
    print(f"\n{'='*80}")
    print("TESTING POSITIONS")
    print(f"{'='*80}")
    
    matches = 0
    mismatches = 0
    
    for i, pos in enumerate(flagged_positions[:5]):
        move_num = pos.get('move_number')
        original_hash = pos.get('sym_hash')
        
        print(f"\n--- Position {i} ---")
        print(f"move_number: {move_num}")
        print(f"original sym_hash: {original_hash}")
        
        # Reconstruct: take first move_num moves from complete game
        if move_num <= len(complete_game):
            reconstruct_moves = complete_game[:move_num]
            print(f"Replaying {len(reconstruct_moves)} moves through KataGo...")
            
            reconstructed_hash = query_katago(
                reconstruct_moves,
                settings.get('komi'),
                settings.get('board_size'),
                settings.get('rules')
            )
            
            if reconstructed_hash:
                print(f"reconstructed sym_hash: {reconstructed_hash}")
                
                if reconstructed_hash == original_hash:
                    print("✓ HASHES MATCH!")
                    matches += 1
                else:
                    print("✗ HASHES DO NOT MATCH")
                    mismatches += 1
            else:
                print("✗ Failed to get hash from KataGo")
                mismatches += 1
        else:
            print(f"✗ Cannot reconstruct - move_num ({move_num}) > complete_game ({len(complete_game)})")
            mismatches += 1
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    
    if matches > 0 and mismatches == 0:
        print("\n✓✓✓ SUCCESS! All reconstructed hashes match!")
        print("This proves reconstruction works when using the complete game sequence.")
    elif matches > 0:
        print(f"\n⚠ Partial success: {matches}/{matches+mismatches} positions matched")
    else:
        print("\n✗ No matches found. Reconstruction strategy may need adjustment.")

if __name__ == "__main__":
    main()
