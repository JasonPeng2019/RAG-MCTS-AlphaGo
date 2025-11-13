"""
Example Integration with KataGo Client
This file shows how to integrate the Phase 1 tuning scripts with your KataGo client.

Replace the placeholder methods in phase1_uncertainty_tuning.py with these implementations.
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

# Import your KataGo client
# from datago.clients.katago_client import KataGoClient


@dataclass
class Position:
    """Represents a game position with uncertainty metrics"""
    board_state: np.ndarray
    policy: np.ndarray
    value: float
    ownership: np.ndarray
    move_number: int
    uncertainty_score: float


class GameExecutor:
    """
    Example integration with KataGo for running games with uncertainty tracking.
    
    This class shows how to:
    1. Run a game with KataGo
    2. Compute uncertainty scores at each position
    3. Store uncertain positions in RAG
    4. Track game statistics
    """
    
    def __init__(self, katago_config_path: str, rag_store=None):
        """
        Initialize the game executor.
        
        Args:
            katago_config_path: Path to KataGo analysis config
            rag_store: RAG storage backend (optional for Phase 1a)
        """
        # Initialize KataGo client
        # self.katago = KataGoClient(katago_config_path)
        self.rag_store = rag_store
    
    def compute_policy_entropy(self, policy: np.ndarray) -> float:
        """
        Compute cross-entropy of policy distribution (E).
        
        High entropy = many similar-probability moves = uncertain
        
        Args:
            policy: Policy distribution array (361 values for 19x19)
            
        Returns:
            Cross-entropy value
        """
        # Remove zero probabilities to avoid log(0)
        policy_nonzero = policy[policy > 1e-10]
        
        # Compute entropy: -sum(p * log(p))
        entropy = -np.sum(policy_nonzero * np.log(policy_nonzero))
        
        return entropy
    
    def compute_value_sparseness(self, value_distribution: Dict[str, float]) -> float:
        """
        Compute sparseness of value distribution (K).
        
        After MCTS backpropagation, KataGo has value estimates for each searched node.
        High sparseness = values are spread out = uncertain position
        
        Args:
            value_distribution: Dictionary of {move: value_estimate}
            
        Returns:
            Sparseness score (higher = more uncertain)
        """
        if not value_distribution:
            return 0.0
        
        values = np.array(list(value_distribution.values()))
        
        # Method 1: Standard deviation
        sparseness = np.std(values)
        
        # Alternative Method 2: Range / mean
        # if len(values) > 1:
        #     sparseness = (np.max(values) - np.min(values)) / (np.mean(np.abs(values)) + 1e-6)
        
        return sparseness
    
    def run_game_with_uncertainty_tracking(self, 
                                          uncertainty_config,
                                          storage_threshold: float = None,
                                          game_id: int = 0) -> Dict:
        """
        Run a single game while tracking uncertainty and storing positions.
        
        This is the method that should replace run_single_game() in phase1_uncertainty_tuning.py
        
        Args:
            uncertainty_config: UncertaintyConfig from phase1_uncertainty_tuning.py
            storage_threshold: If provided, store positions above this threshold
            game_id: Game identifier for logging
            
        Returns:
            Dictionary with game results and statistics
        """
        import time
        start_time = time.time()
        
        # Initialize game
        # game = self.katago.new_game()
        
        positions_analyzed = []
        positions_stored = []
        uncertainty_scores = []
        
        move_number = 0
        game_over = False
        
        # Play game loop
        while not game_over and move_number < 400:  # Max 400 moves
            move_number += 1
            
            # Get KataGo analysis for current position
            # analysis = self.katago.analyze_position(game.get_board_state())
            
            # PLACEHOLDER: Replace with actual KataGo analysis
            # For demonstration, create dummy analysis
            analysis = self._get_dummy_analysis(move_number)
            
            # Extract data from analysis
            policy = analysis['policy']
            value = analysis['value']
            ownership = analysis.get('ownership', np.zeros(361))
            value_distribution = analysis.get('value_distribution', {})
            
            # Compute uncertainty metrics
            policy_entropy = self.compute_policy_entropy(policy)
            value_sparseness = self.compute_value_sparseness(value_distribution)
            
            # Compute uncertainty score using the config
            stones_on_board = move_number  # Approximate
            uncertainty_score = uncertainty_config.compute_uncertainty(
                policy_entropy=policy_entropy,
                value_sparseness=value_sparseness,
                stones_on_board=stones_on_board,
                total_stones=361
            )
            
            uncertainty_scores.append(uncertainty_score)
            
            # Store position if above threshold
            if storage_threshold is not None and uncertainty_score > storage_threshold:
                position = Position(
                    board_state=analysis['board_state'],
                    policy=policy,
                    value=value,
                    ownership=ownership,
                    move_number=move_number,
                    uncertainty_score=uncertainty_score
                )
                positions_stored.append(position)
                
                # Store in RAG database
                if self.rag_store:
                    self.rag_store.store_position(position)
            
            # Make move (use policy to select)
            # move = self.katago.select_move(policy)
            # game.play_move(move)
            
            # Check if game is over
            # game_over = game.is_finished()
            game_over = move_number >= 250  # Placeholder
        
        # Get final result
        # winner = game.get_winner()
        # our_color = game.get_our_color()
        # win = (winner == our_color)
        
        # Placeholder result
        import random
        win = random.random() > 0.5
        
        computation_time = time.time() - start_time
        
        return {
            'game_id': game_id,
            'win': win,
            'game_length': move_number,
            'avg_uncertainty': np.mean(uncertainty_scores) if uncertainty_scores else 0.0,
            'max_uncertainty': np.max(uncertainty_scores) if uncertainty_scores else 0.0,
            'num_stored_positions': len(positions_stored),
            'computation_time': computation_time,
            'uncertainty_scores': uncertainty_scores
        }
    
    def _get_dummy_analysis(self, move_number: int) -> Dict:
        """
        Dummy analysis for testing. Replace with actual KataGo analysis.
        """
        import random
        
        # Generate random policy (should be normalized)
        policy = np.random.dirichlet(np.ones(361) * 0.5)
        
        # Random value
        value = random.uniform(-0.3, 0.3)
        
        # Dummy board state (19x19)
        board_state = np.random.randint(-1, 2, size=(19, 19))
        
        # Dummy value distribution for top moves
        value_distribution = {
            f'move_{i}': random.uniform(-0.2, 0.2)
            for i in range(10)
        }
        
        return {
            'policy': policy,
            'value': value,
            'board_state': board_state,
            'ownership': np.random.uniform(-1, 1, size=361),
            'value_distribution': value_distribution
        }


# Example usage in phase1_uncertainty_tuning.py:
"""
def run_single_game(self, config: UncertaintyConfig, game_id: int) -> GameResult:
    # Initialize executor (do this once in __init__ instead)
    executor = GameExecutor(
        katago_config_path="path/to/katago/analysis.cfg",
        rag_store=self.rag_store  # Pass your RAG store
    )
    
    # Run game with uncertainty tracking
    game_result = executor.run_game_with_uncertainty_tracking(
        uncertainty_config=config,
        storage_threshold=self.storage_threshold,  # Set in Phase 1b
        game_id=game_id
    )
    
    # Convert to GameResult dataclass
    result = GameResult(
        config_id=config.config_id,
        win=game_result['win'],
        game_length=game_result['game_length'],
        avg_uncertainty=game_result['avg_uncertainty'],
        max_uncertainty=game_result['max_uncertainty'],
        num_stored_positions=game_result['num_stored_positions'],
        computation_time=game_result['computation_time']
    )
    
    return result
"""


# Example: Computing uncertainty for a real position
def example_compute_uncertainty():
    """Example showing how to compute uncertainty for a position"""
    from phase1_uncertainty_tuning import UncertaintyConfig
    
    # Load best config from Phase 1
    config = UncertaintyConfig(
        w1=0.5,
        w2=0.5,
        phase_early_multiplier=1.0,
        phase_late_multiplier=1.0
    )
    
    # Example position data (replace with real KataGo analysis)
    policy = np.random.dirichlet(np.ones(361) * 0.5)  # Random policy
    value_dist = {f'move_{i}': np.random.uniform(-0.2, 0.2) for i in range(10)}
    stones_on_board = 50
    
    # Compute metrics
    executor = GameExecutor("dummy_config")
    policy_entropy = executor.compute_policy_entropy(policy)
    value_sparseness = executor.compute_value_sparseness(value_dist)
    
    # Compute final uncertainty score
    uncertainty = config.compute_uncertainty(
        policy_entropy=policy_entropy,
        value_sparseness=value_sparseness,
        stones_on_board=stones_on_board
    )
    
    print(f"Policy entropy: {policy_entropy:.4f}")
    print(f"Value sparseness: {value_sparseness:.4f}")
    print(f"Uncertainty score: {uncertainty:.4f}")
    
    return uncertainty


if __name__ == "__main__":
    example_compute_uncertainty()
