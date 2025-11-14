import json
import subprocess
import csv
from typing import Optional, List

try:
    from sgfmill import boards
except ImportError:
    boards = None


def parse_flagged_positions_csv(csv_path: str) -> List[List[List]]:
    """
    Parse a CSV file containing JSON objects and extract all moves_history lists.

    Args:
        csv_path: Path to CSV file where each row contains a JSON object

    Returns:
        List of moves_history lists, where each moves_history is a list of
        [player, location] pairs like [["B","Q4"], ["W","D4"], ...]

    Example:
        >>> moves_histories = parse_flagged_positions_csv("games.csv")
        >>> for moves in moves_histories:
        >>>     print(f"Found game with {len(moves)} moves")
    """
    all_moves_histories = []

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            # Skip empty rows
            if not row:
                continue

            # Assume the JSON is in the first column (adjust if needed)
            json_str = row[0]

            try:
                game_data = json.loads(json_str)

                # Extract flagged_positions field
                if 'flagged_positions' not in game_data:
                    continue

                flagged_positions = game_data['flagged_positions']

                # Extract moves_history from each flagged position
                for position in flagged_positions:
                    if 'moves_history' in position:
                        all_moves_histories.append(position['moves_history'])

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON in row: {e}")
                continue

    return all_moves_histories


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

    json_example = {} # Placeholder for actual JSON input
    flagged_positions = json_example
    
    
    # Analyze a position
    moves = [
        ["B", "Q4"],
        ["W", "D4"],
        ["B", "Q16"],
        ["W", "D16"],
    ]


    
    embedding = analyzer.analyze_position(
        moves=moves,
        komi=7.5,
        rules="chinese",
        max_visits=100  # Quick analysis
    )
    
    # Print results
    print(f"State Hash: {embedding.state_hash}")
    print(f"Sym Hash: {embedding.sym_hash}")
    print(f"Query ID: {embedding.query_id}")
    print(f"Winrate: {embedding.winrate:.3f}")
    print(f"Score Lead: {embedding.score_lead:.2f}")
    print(f"Komi: {embedding.komi}")
    print(f"Policy shape: {len(embedding.policy) if embedding.policy else 'None'}")
    print(f"Ownership shape: {len(embedding.ownership) if embedding.ownership else 'None'}")
    print(f"Move infos: {len(embedding.move_infos) if embedding.move_infos else 'None'}")
    
    # Convert to dict for storage
    data = embedding.to_dict()
    print(f"\nStorable dict keys: {list(data.keys())}")
    
    analyzer.close()