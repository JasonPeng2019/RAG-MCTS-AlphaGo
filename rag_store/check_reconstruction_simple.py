#!/usr/bin/env python3
"""
Simple count of positions reconstructed from rag_data_* folders.
Shows how many source positions existed vs how many were reconstructed.
"""

import json
import os
from pathlib import Path

def count_source_positions(rag_data_dir):
    """Count total flagged positions in source rag_data directory."""
    total = 0
    file_count = 0
    
    if not os.path.exists(rag_data_dir):
        return 0, 0
    
    json_files = list(Path(rag_data_dir).glob("RAG_rawdata_game_*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            file_count += 1
            
            if 'flagged_positions' in data:
                total += len(data['flagged_positions'])
        except:
            pass
    
    return file_count, total

def count_reconstructed_positions(output_dir):
    """Count reconstructed positions in rag_database.json."""
    db_path = os.path.join(output_dir, "rag_database.json")
    
    if not os.path.exists(db_path):
        return 0
    
    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return len(data)
    except:
        pass
    
    return 0

def main():
    base_dir = Path(__file__).parent
    build_dir = base_dir / "../../build"
    
    print("="*90)
    print("RAG Data Reconstruction Summary")
    print("="*90)
    print()
    print(f"{'Set':<6} {'Source Files':<14} {'Source Positions':<18} {'Reconstructed':<15} {'Coverage':<10}")
    print("-"*90)
    
    total_files = 0
    total_source = 0
    total_reconstructed = 0
    
    for i in range(1, 6):
        rag_data_dir = build_dir / f"rag_data_{i}"
        output_dir = base_dir / f"rag_output_{i}_done"
        
        files, source_pos = count_source_positions(rag_data_dir)
        recon_pos = count_reconstructed_positions(output_dir)
        
        coverage = (recon_pos / source_pos * 100) if source_pos > 0 else 0
        
        print(f"{i:<6} {files:<14,} {source_pos:<18,} {recon_pos:<15,} {coverage:>6.2f}%")
        
        total_files += files
        total_source += source_pos
        total_reconstructed += recon_pos
    
    print("-"*90)
    total_coverage = (total_reconstructed / total_source * 100) if total_source > 0 else 0
    print(f"{'TOTAL':<6} {total_files:<14,} {total_source:<18,} {total_reconstructed:<15,} {total_coverage:>6.2f}%")
    print("="*90)
    print()
    print("EXPLANATION:")
    print(f"  • Total source positions flagged:  {total_source:,}")
    print(f"  • Total positions reconstructed:   {total_reconstructed:,}")
    print(f"  • Reconstruction rate:              {total_coverage:.2f}%")
    print()
    print("NOTE: Sym_hash values differ between source and reconstructed data because")
    print("game_analyzer.py performs fresh KataGo analysis, generating new hashes.")
    print("The low coverage indicates only a subset of flagged positions were reconstructed.")
    print("="*90)

if __name__ == "__main__":
    main()
