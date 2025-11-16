import json
import os
from pathlib import Path
from typing import Dict, List

def load_all_positions_from_directory(directory: str) -> Dict[str, dict]:
    positions = {}
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Warning: Directory not found: {directory}")
        return positions
    
    json_files = list(dir_path.glob("*.json"))
    print(f"Loading {len(json_files)} JSON files from {directory}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different formats
            if 'flagged_positions' in data:
                # Shallow format (from rag_data)
                for pos in data['flagged_positions']:
                    sym_hash = pos.get('sym_hash')
                    if sym_hash:
                        positions[sym_hash] = pos
            elif isinstance(data, list):
                # Deep format (from rag_output - array of positions)
                for pos in data:
                    sym_hash = pos.get('sym_hash')
                    if sym_hash:
                        positions[sym_hash] = pos
            else:
                print(f"Warning: Unknown format in {json_file}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"  Loaded {len(positions)} unique positions")
    return positions


def create_matched_pairs(shallow_dirs: List[str], deep_dirs: List[str], output_path: str):
    """
    Create matched training data by aligning shallow and deep positions.
    
    Args:
        shallow_dirs: List of directories containing shallow (selfplay) data
        deep_dirs: List of directories containing deep (reconstructed) data
        output_path: Path to save matched_training_data.json
    """
    print("="*60)
    print("Creating Matched Training Data")
    print("="*60)
    
    # Load all shallow positions
    print("\n1. Loading shallow (selfplay) positions...")
    shallow_positions = {}
    for shallow_dir in shallow_dirs:
        shallow_positions.update(load_all_positions_from_directory(shallow_dir))
    
    print(f"\nTotal shallow positions: {len(shallow_positions)}")
    
    # Load all deep positions
    print("\n2. Loading deep (reconstructed) positions...")
    deep_positions = {}
    for deep_dir in deep_dirs:
        deep_positions.update(load_all_positions_from_directory(deep_dir))
    
    print(f"\nTotal deep positions: {len(deep_positions)}")
    
    # Match positions by sym_hash
    print("\n3. Matching positions by sym_hash...")
    matched_pairs = []
    
    for sym_hash, shallow_data in shallow_positions.items():
        if sym_hash in deep_positions:
            matched_pairs.append({
                'sym_hash': sym_hash,
                'shallow': shallow_data,
                'deep': deep_positions[sym_hash]
            })
    
    print(f"\nMatched {len(matched_pairs)} position pairs")
    
    # Calculate statistics
    shallow_only = len(shallow_positions) - len(matched_pairs)
    deep_only = len(deep_positions) - len(matched_pairs)
    
    print(f"\nStatistics:")
    print(f"  Matched pairs: {len(matched_pairs)}")
    print(f"  Shallow only (no deep match): {shallow_only}")
    print(f"  Deep only (no shallow match): {deep_only}")
    print(f"  Match rate: {len(matched_pairs)/len(shallow_positions)*100:.1f}%")
    
    # Save as TWO separate files for phase1_uncertainty_tuning.py
    print(f"\n4. Saving matched data...")
    
    # Create shallow database (list of shallow positions)
    shallow_db = [pair['shallow'] for pair in matched_pairs]
    shallow_path = output_path.replace('.json', '_shallow.json')
    
    with open(shallow_path, 'w', encoding='utf-8') as f:
        json.dump(shallow_db, f, indent=2)
    
    # Create deep database (list of deep positions)
    deep_db = [pair['deep'] for pair in matched_pairs]
    deep_path = output_path.replace('.json', '_deep.json')
    
    with open(deep_path, 'w', encoding='utf-8') as f:
        json.dump(deep_db, f, indent=2)
    
    shallow_size_mb = os.path.getsize(shallow_path) / 1024 / 1024
    deep_size_mb = os.path.getsize(deep_path) / 1024 / 1024
    
    print(f"✓ Successfully created matched training data")
    print(f"  Shallow DB: {shallow_path}")
    print(f"    - Size: {shallow_size_mb:.2f} MB")
    print(f"    - Positions: {len(shallow_db)}")
    print(f"  Deep DB: {deep_path}")
    print(f"    - Size: {deep_size_mb:.2f} MB")
    print(f"    - Positions: {len(deep_db)}")
    print(f"\nNote: Positions are aligned by index (same sym_hash at each index)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create matched training data for phase1 tuning')
    parser.add_argument('--output', default='./matched_training_data.json',
                       help='Output path for matched data JSON')
    
    args = parser.parse_args()
    
    # Define directories - search in BOTH current and _done folders
    shallow_dirs = [
        '../../build/rag_data_1',
        '../../build/rag_data_1_done',
        '../../build/rag_data_2',
        '../../build/rag_data_2_done',
        '../../build/rag_data_3',
        '../../build/rag_data_3_done',
        '../../build/rag_data_4',
        '../../build/rag_data_4_done',
        '../../build/rag_data_5',
        '../../build/rag_data_5_done'
    ]
    
    deep_dirs = [
        './rag_output_1_done',
        './rag_output_2_done',
        './rag_output_3_done',
        './rag_output_4_done',
        './rag_output_5_done'
    ]
    
    create_matched_pairs(shallow_dirs, deep_dirs, args.output)
