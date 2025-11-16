#!/usr/bin/env python3
"""
Check how many positions from rag_data_* were successfully reconstructed
in rag_output_*_done by comparing sym_hash values.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_sym_hashes_from_rag_data(rag_data_dir):
    """Load all sym_hashes from RAG data JSON files."""
    sym_hashes = set()
    file_count = 0
    position_count = 0
    
    print(f"Loading sym_hashes from {rag_data_dir}...")
    
    if not os.path.exists(rag_data_dir):
        print(f"  ⚠️  Directory not found: {rag_data_dir}")
        return sym_hashes, file_count, position_count
    
    json_files = list(Path(rag_data_dir).glob("RAG_rawdata_game_*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            file_count += 1
            
            # Check for flagged_positions in the game data
            if 'flagged_positions' in data:
                for pos in data['flagged_positions']:
                    if 'sym_hash' in pos:
                        sym_hashes.add(pos['sym_hash'])
                        position_count += 1
                    elif 'symHash' in pos:
                        sym_hashes.add(pos['symHash'])
                        position_count += 1
            
        except Exception as e:
            print(f"  Error reading {json_file.name}: {e}")
    
    print(f"  Files: {file_count}, Positions: {position_count}, Unique sym_hashes: {len(sym_hashes)}")
    return sym_hashes, file_count, position_count

def load_sym_hashes_from_output(output_dir):
    """Load all sym_hashes from reconstructed rag_database.json."""
    sym_hashes = set()
    position_count = 0
    
    db_path = os.path.join(output_dir, "rag_database.json")
    
    print(f"Loading sym_hashes from {output_dir}...")
    
    if not os.path.exists(db_path):
        print(f"  ⚠️  Database not found: {db_path}")
        return sym_hashes, position_count
    
    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
        
        # The database should be a list of analyzed positions
        if isinstance(data, list):
            for pos in data:
                if 'sym_hash' in pos:
                    sym_hashes.add(pos['sym_hash'])
                    position_count += 1
                elif 'symHash' in pos:
                    sym_hashes.add(pos['symHash'])
                    position_count += 1
        
        print(f"  Positions: {position_count}, Unique sym_hashes: {len(sym_hashes)}")
    
    except Exception as e:
        print(f"  Error reading database: {e}")
    
    return sym_hashes, position_count

def main():
    base_dir = Path(__file__).parent
    build_dir = base_dir / "../../build"
    
    print("="*80)
    print("RAG Data Reconstruction Coverage Analysis")
    print("="*80)
    print()
    
    # Process each of the 5 parallel reconstruction sets
    total_source_hashes = set()
    total_reconstructed_hashes = set()
    
    stats = []
    
    for i in range(1, 6):
        print(f"\n{'='*80}")
        print(f"SET {i}: rag_data_{i} -> rag_output_{i}_done")
        print(f"{'='*80}")
        
        # Load source data
        rag_data_dir = build_dir / f"rag_data_{i}"
        source_hashes, source_files, source_positions = load_sym_hashes_from_rag_data(rag_data_dir)
        
        # Load reconstructed data
        output_dir = base_dir / f"rag_output_{i}_done"
        recon_hashes, recon_positions = load_sym_hashes_from_output(output_dir)
        
        # Calculate coverage
        matched = source_hashes & recon_hashes
        missing = source_hashes - recon_hashes
        extra = recon_hashes - source_hashes
        
        coverage_pct = (len(matched) / len(source_hashes) * 100) if source_hashes else 0
        
        print(f"\nResults:")
        print(f"  Source positions:       {len(source_hashes):,}")
        print(f"  Reconstructed:          {len(recon_hashes):,}")
        print(f"  Matched (found):        {len(matched):,}")
        print(f"  Missing (not found):    {len(missing):,}")
        print(f"  Extra (unexpected):     {len(extra):,}")
        print(f"  Coverage:               {coverage_pct:.2f}%")
        
        stats.append({
            'set': i,
            'source_files': source_files,
            'source_positions': len(source_hashes),
            'reconstructed': len(recon_hashes),
            'matched': len(matched),
            'missing': len(missing),
            'extra': len(extra),
            'coverage_pct': coverage_pct
        })
        
        # Add to totals
        total_source_hashes.update(source_hashes)
        total_reconstructed_hashes.update(recon_hashes)
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY (All 5 Sets Combined)")
    print(f"{'='*80}")
    
    total_matched = total_source_hashes & total_reconstructed_hashes
    total_missing = total_source_hashes - total_reconstructed_hashes
    total_extra = total_reconstructed_hashes - total_source_hashes
    
    total_coverage_pct = (len(total_matched) / len(total_source_hashes) * 100) if total_source_hashes else 0
    
    print(f"\nTotal unique source positions:       {len(total_source_hashes):,}")
    print(f"Total unique reconstructed:          {len(total_reconstructed_hashes):,}")
    print(f"Total matched:                       {len(total_matched):,}")
    print(f"Total missing:                       {len(total_missing):,}")
    print(f"Total extra:                         {len(total_extra):,}")
    print(f"Overall coverage:                    {total_coverage_pct:.2f}%")
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Set':<6} {'Files':<8} {'Source':<10} {'Recon':<10} {'Matched':<10} {'Coverage':<10}")
    print(f"{'-'*80}")
    
    total_files = 0
    total_source = 0
    total_recon = 0
    total_match = 0
    
    for s in stats:
        print(f"{s['set']:<6} {s['source_files']:<8} {s['source_positions']:<10,} "
              f"{s['reconstructed']:<10,} {s['matched']:<10,} {s['coverage_pct']:<10.2f}%")
        total_files += s['source_files']
        total_source += s['source_positions']
        total_recon += s['reconstructed']
        total_match += s['matched']
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<6} {total_files:<8} {total_source:<10,} {total_recon:<10,} "
          f"{total_match:<10,} {(total_match/total_source*100 if total_source else 0):<10.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Note: Source counts may include duplicates across sets.")
    print(f"Overall stats show unique positions across all sets.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
