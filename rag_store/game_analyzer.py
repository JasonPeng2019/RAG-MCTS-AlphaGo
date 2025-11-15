import json
import subprocess
import csv
import os
from pathlib import Path
from typing import Optional, List, Dict

try:
    from sgfmill import boards
except ImportError:
    boards = None


def parse_flagged_positions_csv(csv_path: str, json_dir: str) -> List[Dict]:
    """
    Parse a CSV file containing JSON filenames, load those JSON files from a directory,
    and extract all flagged positions with their moves_history.

    Args:
        csv_path: Path to CSV file where each row contains a JSON filename
        json_dir: Directory path where the JSON game files are stored

    Returns:
        List of dictionaries, each containing:
        - 'game_id': The game identifier
        - 'filename': The JSON filename
        - 'moves_history': List of [player, location] pairs like [["B","Q4"], ["W","D4"], ...]
        - 'position_data': Full flagged position data including uncertainty metrics, children, etc.

    Example:
        >>> positions = parse_flagged_positions_csv("games.csv", "/path/to/rag_data")
        >>> for pos in positions:
        >>>     print(f"Game {pos['game_id']}: {len(pos['moves_history'])} moves")
    """
    all_flagged_positions = []
    json_dir_path = Path(json_dir)

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Skip header if present
        header = next(csv_reader, None)

        for row in csv_reader:
            # Skip empty rows
            if not row:
                continue

            # Get the JSON filename from the first column
            json_filename = row[0].strip()
            
            # Construct full path to JSON file
            json_path = json_dir_path / json_filename
            
            if not json_path.exists():
                print(f"Warning: JSON file not found: {json_path}")
                continue

            try:
                # Load the JSON game file
                with open(json_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)

                # Extract game_id
                game_id = game_data.get('game_id', 'unknown')

                # Extract flagged_positions field
                if 'flagged_positions' not in game_data:
                    print(f"Warning: No flagged_positions in {json_filename}")
                    continue

                flagged_positions = game_data['flagged_positions']

                # Extract each flagged position with its full data
                for position in flagged_positions:
                    if 'moves_history' in position:
                        all_flagged_positions.append({
                            'game_id': game_id,
                            'filename': json_filename,
                            'moves_history': position['moves_history'],
                            'position_data': position  # Include full position data
                        })

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON file {json_filename}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing {json_filename}: {e}")
                continue

    return all_flagged_positions


def count_stones_on_board(moves: list, board_size: int = 19) -> dict:
    """
    Count stones on board by replaying moves with capture logic.
    Call this on-demand when retrieving from RAG, not during storage.

    Args:
        moves: List of [player, location] like [["B","Q4"], ["W","D4"]]
        board_size: Board size (default 19)

    Returns:
        {'black': int, 'white': int, 'total': int}
    """
    if boards is None:
        raise ImportError("sgfmill not installed. Run: pip install sgfmill")

    board = boards.Board(board_size)

    for player, location in moves:
        if location.upper() == 'PASS':
            continue

        # Parse KataGo coordinate format (e.g., "Q4")
        col = ord(location[0].upper()) - ord('A')
        if col >= 8:  # Skip 'I' in Go coordinates
            col -= 1
        row = board_size - int(location[1:])

        # Play move (sgfmill handles captures automatically)
        color = 'b' if player == 'B' else 'w'
        board.play(row, col, color)

    # Count stones
    black_count = 0
    white_count = 0
    for r in range(board_size):
        for c in range(board_size):
            stone = board.get(r, c)
            if stone == 'b':
                black_count += 1
            elif stone == 'w':
                white_count += 1

    return {
        'black': black_count,
        'white': white_count,
        'total': black_count + white_count
    }


class GoStateEmbedding:
    """Stores KataGo state embedding for RAG storage."""

    def __init__(self, katago_response: dict, query_info: dict):
        # Stored vectors
        self.state_hash = katago_response['rootInfo']['thisHash']
        self.sym_hash = katago_response['rootInfo']['symHash']
        self.policy = katago_response.get('policy', None)
        self.ownership = katago_response.get('ownership', None)
        self.winrate = katago_response['rootInfo']['winrate']
        self.score_lead = katago_response['rootInfo']['scoreLead']
        self.move_infos = katago_response.get('moveInfos', None)
        self.komi = query_info['komi']
        self.query_id = katago_response['id']

        # Temporary fields for storage decision (not saved to DB)
        self.score_stdev = katago_response['rootInfo']['scoreStdev']
        self.lcb = katago_response['rootInfo']['lcb']

    def to_dict(self):
        """Convert to dictionary for RAG storage."""
        return {
            'sym_hash': self.sym_hash,
            'state_hash': self.state_hash,
            'policy': self.policy,
            'ownership': self.ownership,
            'winrate': self.winrate,
            'score_lead': self.score_lead,
            'move_infos': self.move_infos,
            'komi': self.komi,
            'query_id': self.query_id,
        }

class KataGoAnalyzer:
    """Wrapper for KataGo analysis engine"""
    
    def __init__(self, katago_path: str, config_path: str, model_path: str):
        self.katago = subprocess.Popen(
            [katago_path, 'analysis', '-config', config_path, '-model', model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.query_counter = 0
    
    def analyze_position(self, moves: list, komi: float = 7.5, 
                        rules: str = "chinese",
                        board_size: int = 19,
                        max_visits: Optional[int] = None) -> GoStateEmbedding:
        """
        Analyze a position and return embedding.
        
        Args:
            moves: List of [player, location] like [["B","Q4"], ["W","D4"]]
            komi: Game komi
            rules: Rule set (chinese, japanese, tromp-taylor, etc.)
            board_size: Board size (default 19x19)
            max_visits: Optional visit limit
        
        Returns:
            GoStateEmbedding with all extracted features
        """
        
        query = {
            "id": f"query_{self.query_counter}",
            "moves": moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "includePolicy": True,
            "includeOwnership": True,
        }

        if max_visits:
            query["maxVisits"] = max_visits

        self.query_counter += 1

        self.katago.stdin.write(json.dumps(query) + '\n')
        self.katago.stdin.flush()

        response_line = self.katago.stdout.readline()
        response = json.loads(response_line)

        if 'error' in response:
            raise RuntimeError(f"KataGo error: {response['error']}")

        return GoStateEmbedding(response, query)
    
    def close(self):
        self.katago.stdin.close()
        self.katago.wait()


# Example Usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = KataGoAnalyzer(
        katago_path="./katago",
        config_path="cpp/configs/analysis_example.cfg",
        model_path="cpp/tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz"
    )

    # Configure paths
    csv_path = "rag_files_list.csv"  # CSV containing JSON filenames
    json_dir = "../../build/rag_data"  # Directory containing the JSON game files
    output_dir = "./json"  # Directory to save output JSON files

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parse CSV and load all flagged positions from JSON files
    flagged_positions = parse_flagged_positions_csv(csv_path, json_dir)
    
    print(f"Found {len(flagged_positions)} flagged positions from CSV")

    for idx, position_info in enumerate(flagged_positions):
        output_json = {}
        game_id = position_info['game_id']
        filename = position_info['filename']
        moves_history = position_info['moves_history']
        position_data = position_info['position_data']
        position_children = position_info['children']
        
        print(f"\n{'='*60}")
        print(f"Processing position {idx+1}/{len(flagged_positions)}")
        print(f"Game ID: {game_id}")
        print(f"Filename: {filename}")
        print(f"Move number: {position_data.get('move_number', 'N/A')}")
        print(f"Moves played: {len(moves_history)}")
        
        # Analyze position with KataGo
        embedding = analyzer.analyze_position(
            moves=moves_history,
            komi=7.5,
            rules="chinese",
            max_visits=5  # Quick analysis
        ) 

        # Print results
        output_json['state_hash'] = embedding.state_hash
        output_json['sym_hash'] = embedding.sym_hash
        output_json['policy'] = embedding.policy
        output_json['ownership'] = embedding.ownership
        output_json['winrate'] = embedding.winrate
        output_json['score_lead'] = embedding.score_lead
        output_json['move_infos'] = embedding.move_infos
        output_json['komi'] = embedding.komi
        output_json['query_id'] = embedding.query_id
        output_json['stone_count'] = count_stones_on_board(moves_history)
        for child in position_children:
            output_json['child_nodes'] = {child['move']: [child['value'], child['policy']]}

        # Save output_json to file
        output_filename = f"{game_id}_position_{idx+1}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4)

        print(f"Saved JSON to: {output_path}")

        # Print uncertainty metrics from original data
        if 'uncertainty_metrics' in position_data:
            metrics = position_data['uncertainty_metrics']
            print(f"Policy Entropy: {metrics.get('policy_entropy', 'N/A')}")
            print(f"Value Variance: {metrics.get('value_variance', 'N/A')}")
            print(f"Combined Score: {metrics.get('combined_score', 'N/A')}")
        
        # Convert to dict for storage
        data = embedding.to_dict()
        print(f"Storable dict keys: {list(data.keys())}")
    
    analyzer.close()