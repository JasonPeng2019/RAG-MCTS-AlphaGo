#!/usr/bin/env python3
"""
Find actual correlation between rag_data_* source and rag_output_*_done reconstructed data.
Ignores CSV files - compares actual data content.
"""

import json
from pathlib import Path
from collections import defaultdict

def get_position_signature(moves_history, komi, board_size):
    """Create a signature for matching positions across datasets."""
    # Use first few moves + game metadata as signature
    moves_key = tuple(tuple(m) for m in moves_history[:10]) if moves_history else ()
    return (moves_key, komi, board_size)

def main():
    base_dir = Path(__file__).parent
    build_dir = base_dir / "../../build"
    
    print("="*80)
    print("ACTUAL CORRELATION CHECK (Ignoring CSV files)")
    print("="*80)
    
    for set_num in range(1, 6):
        print(f"\n{'='*80}")
        print(f"SET {set_num}: rag_data_{set_num} vs rag_output_{set_num}_done")
        print(f"{'='*80}")
        
        # Load ALL source positions from rag_data_*
        print(f"\nLoading source positions from rag_data_{set_num}...")
        source_dir = build_dir / f"rag_data_{set_num}"
        source_signatures = {}  # signature -> (filename, position_info)
        source_files_processed = 0
        total_source_positions = 0
        
        for json_file in source_dir.glob("RAG_rawdata_game_*.json"):
            source_files_processed += 1
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                game_id = data.get('game_id', 'unknown')
                settings = data.get('settings', {})
                komi = settings.get('komi', 0)
                board_size = settings.get('board_size', 19)
                
                for pos in data.get('flagged_positions', []):
                    total_source_positions += 1
                    moves = pos.get('moves_history', [])
                    sig = get_position_signature(moves, komi, board_size)
                    
                    if sig not in source_signatures:
                        source_signatures[sig] = {
                            'filename': json_file.name,
                            'game_id': game_id,
                            'move_number': pos.get('move_number'),
                            'source_sym_hash': pos.get('sym_hash'),
                            'moves_count': len(moves)
                        }
            except Exception as e:
                print(f"  Error reading {json_file.name}: {e}")
        
        print(f"  Files: {source_files_processed}")
        print(f"  Total positions: {total_source_positions}")
        print(f"  Unique signatures: {len(source_signatures)}")
        
        # Load reconstructed positions
        print(f"\nLoading reconstructed positions from rag_output_{set_num}_done...")
        output_dir = base_dir / f"rag_output_{set_num}_done"
        db_path = output_dir / "rag_database.json"
        
        if not db_path.exists():
            print(f"  ⚠️  Database not found")
            continue
        
        with open(db_path) as f:
            reconstructed_data = json.load(f)
        
        print(f"  Total reconstructed: {len(reconstructed_data)}")
        
        # Try to match reconstructed to source by signature
        matches = 0
        no_match = 0
        recon_signatures = {}
        
        for idx, recon_pos in enumerate(reconstructed_data):
            # Reconstructed positions don't have moves_history stored!
            # They only have the analysis results
            # We need to find another way to correlate...
            
            komi = recon_pos.get('komi')
            stone_count = recon_pos.get('stone_count', {}).get('total', 0)
            
            # Store for analysis
            recon_signatures[idx] = {
                'komi': komi,
                'stone_count': stone_count,
                'sym_hash': recon_pos.get('sym_hash')
            }
        
        # Check komi distribution to see if they're from same games
        source_komis = defaultdict(int)
        recon_komis = defaultdict(int)
        
        for sig, info in source_signatures.items():
            komi = sig[1]
            source_komis[komi] += 1
        
        for idx, info in recon_signatures.items():
            komi = info['komi']
            recon_komis[komi] += 1
        
        print(f"\n{'='*80}")
        print(f"KOMI DISTRIBUTION COMPARISON")
        print(f"{'='*80}")
        print(f"{'Komi':<10} {'Source':<15} {'Reconstructed':<15} {'Match?':<10}")
        print(f"{'-'*80}")
        
        all_komis = sorted(set(source_komis.keys()) | set(recon_komis.keys()))
        matching_komis = 0
        
        for komi in all_komis[:10]:  # Show first 10
            source_count = source_komis.get(komi, 0)
            recon_count = recon_komis.get(komi, 0)
            match = "✓" if source_count > 0 and recon_count > 0 else "✗"
            if source_count > 0 and recon_count > 0:
                matching_komis += 1
            print(f"{komi:<10} {source_count:<15} {recon_count:<15} {match:<10}")
        
        if len(all_komis) > 10:
            print(f"... and {len(all_komis) - 10} more komi values")
        
        print(f"\n{'='*80}")
        print(f"SET {set_num} SUMMARY")
        print(f"{'='*80}")
        print(f"Source files: {source_files_processed}")
        print(f"Source positions: {total_source_positions}")
        print(f"Reconstructed positions: {len(reconstructed_data)}")
        print(f"Komi values in source: {len(source_komis)}")
        print(f"Komi values in reconstructed: {len(recon_komis)}")
        print(f"Matching komi values: {matching_komis}")
        
        # Correlation assessment
        if matching_komis > 0:
            print(f"\n✓ CORRELATION FOUND: {matching_komis} komi values appear in both datasets")
        else:
            print(f"\n✗ NO CORRELATION: No matching komi values found")
    
    print(f"\n{'='*80}")
    print("NOTE: Komi matching shows if games came from same source.")
    print("Unusual komi values (like 45, -29, 41.5) are rare, so matches = correlation.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
