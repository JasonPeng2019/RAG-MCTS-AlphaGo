"""
gomill_player.py

Gomill integration for DataGo bot to play games through the gomill library.

This module provides a wrapper that allows the DataGo bot to interface with
gomill for playing games against other engines (like KataGo) or for
tournament play.

Gomill is a Python library for playing Go games and running tournaments.
See: https://github.com/mattheww/gomill
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple
import numpy as np

try:
    from gomill import ascii_boards, boards, gtp_controller, gtp_states
except ImportError:
    raise ImportError(
        "gomill library not found. Install with: pip install gomill"
    )

from .datago_bot import DataGoBot, GameState

logger = logging.getLogger(__name__)


class GomillPlayer:
    """
    Wrapper that allows DataGo bot to play through gomill.
    
    This class implements the interface expected by gomill for a Go playing engine,
    delegating actual move generation to the DataGoBot.
    """
    
    def __init__(self, bot: DataGoBot, color: str = "b"):
        """
        Initialize gomill player wrapper.
        
        Args:
            bot: DataGoBot instance
            color: Player color - "b" for black, "w" for white
        """
        self.bot = bot
        self.color = color
        self.player_int = 1 if color == "b" else -1
        
        # Game tracking
        self.board: Optional[boards.Board] = None
        self.move_history: list[Tuple[str, str]] = []  # (color, move) tuples
        
        logger.info(f"GomillPlayer initialized as {color}")
    
    def setup_game(self, board_size: int = 19, komi: float = 7.5):
        """
        Set up a new game.
        
        Args:
            board_size: Size of the board
            komi: Komi value
        """
        self.board = boards.Board(board_size)
        self.move_history = []
        
        # Initialize bot's game state
        self.bot.new_game(
            board_size=board_size,
            komi=komi,
            player_color=self.player_int,
        )
        
        logger.info(f"Game set up: {board_size}x{board_size}, komi={komi}")
    
    def handle_move(self, color: str, move: Tuple[int, int] | str):
        """
        Handle a move played on the board (by either player).
        
        Args:
            color: "b" or "w"
            move: Either (row, col) tuple or "pass"/"resign"
        """
        if isinstance(move, str):
            move_str = move.lower()
        else:
            # Convert to GTP format
            row, col = move
            move_str = self._coords_to_gtp(row, col, self.board.side)
        
        # Record in history
        self.move_history.append((color, move_str))
        
        # Update gomill board
        if move_str not in ["pass", "resign"]:
            row, col = self._gtp_to_coords(move_str, self.board.side)
            color_int = 1 if color == "b" else -1
            self.board.play(row, col, color_int)
        
        # Update bot's game state
        self._sync_bot_state()
        
        logger.debug(f"Handled move: {color} {move_str}")
    
    def genmove(self) -> str:
        """
        Generate a move for this player.
        
        Returns:
            Move in GTP format (e.g., "D4", "pass", "resign")
        """
        if self.board is None:
            raise RuntimeError("Game not set up. Call setup_game() first.")
        
        logger.info(f"Generating move for {self.color} (move {len(self.move_history)})")
        
        # Generate move using DataGo bot
        decision = self.bot.generate_move()
        move_str = decision.move
        
        # Handle the move
        self.handle_move(self.color, move_str)
        
        # Log decision info
        logger.info(
            f"Generated move: {move_str} "
            f"(unc={decision.uncertainty:.3f}, "
            f"rag={'hit' if decision.rag_hit else 'miss' if decision.rag_queried else 'skip'}, "
            f"time={decision.time_taken_ms:.1f}ms)"
        )
        
        return move_str
    
    def _sync_bot_state(self):
        """Synchronize bot's internal game state with gomill board."""
        if self.board is None:
            return
        
        # Convert gomill board to numpy array
        board_size = self.board.side
        board_array = np.zeros((board_size, board_size), dtype=int)
        
        for row in range(board_size):
            for col in range(board_size):
                color = self.board.get(row, col)
                if color == 'b':
                    board_array[row, col] = 1
                elif color == 'w':
                    board_array[row, col] = -1
        
        # Update bot's game state
        self.bot.game_state.board = board_array
        self.bot.game_state.move_number = len(self.move_history)
        self.bot.game_state.history = self.move_history.copy()
        
        # Update current player
        if len(self.move_history) % 2 == 0:
            self.bot.game_state.current_player = 1  # Black
        else:
            self.bot.game_state.current_player = -1  # White
    
    @staticmethod
    def _coords_to_gtp(row: int, col: int, board_size: int) -> str:
        """
        Convert board coordinates to GTP format.
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            board_size: Size of board
            
        Returns:
            GTP format string (e.g., "D4")
        """
        # GTP uses letters A-T (skipping I) for columns
        col_letters = "ABCDEFGHJKLMNOPQRST"
        if col >= len(col_letters):
            raise ValueError(f"Column {col} out of range")
        
        # GTP rows are numbered from bottom (1) to top (board_size)
        gtp_row = board_size - row
        gtp_col = col_letters[col]
        
        return f"{gtp_col}{gtp_row}"
    
    @staticmethod
    def _gtp_to_coords(move_str: str, board_size: int) -> Tuple[int, int]:
        """
        Convert GTP format to board coordinates.
        
        Args:
            move_str: GTP format string (e.g., "D4")
            board_size: Size of board
            
        Returns:
            (row, col) tuple (0-based)
        """
        move_str = move_str.upper().strip()
        
        if move_str in ["PASS", "RESIGN"]:
            raise ValueError(f"Cannot convert {move_str} to coordinates")
        
        col_letters = "ABCDEFGHJKLMNOPQRST"
        col_char = move_str[0]
        row_str = move_str[1:]
        
        col = col_letters.index(col_char)
        gtp_row = int(row_str)
        row = board_size - gtp_row
        
        return row, col
    
    def get_board_string(self) -> str:
        """Get ASCII representation of current board."""
        if self.board is None:
            return "No game in progress"
        return ascii_boards.render_board(self.board)
    
    def get_statistics(self) -> dict:
        """Get bot statistics for current game."""
        return self.bot.get_statistics()


def play_game_vs_katago(
    datago_config_path: str,
    katago_executable: str,
    katago_model: str,
    katago_config: str,
    board_size: int = 19,
    komi: float = 7.5,
    datago_color: str = "b",
    save_sgf: Optional[str] = None,
) -> dict:
    """
    Play a game between DataGo bot and KataGo using gomill.
    
    Args:
        datago_config_path: Path to DataGo bot config file
        katago_executable: Path to KataGo executable
        katago_model: Path to KataGo model file
        katago_config: Path to KataGo config file
        board_size: Board size
        komi: Komi value
        datago_color: "b" for black, "w" for white
        save_sgf: Optional path to save SGF file
        
    Returns:
        Dictionary with game results and statistics
    """
    logger.info("Starting game: DataGo vs KataGo")
    
    # Initialize DataGo bot
    datago_bot = DataGoBot(datago_config_path)
    datago_player = GomillPlayer(datago_bot, color=datago_color)
    datago_player.setup_game(board_size, komi)
    
    # Initialize KataGo through GTP
    katago_cmd = [
        katago_executable,
        'gtp',
        '-model', katago_model,
        '-config', katago_config,
    ]
    
    try:
        katago_controller = gtp_controller.Gtp_controller(
            katago_cmd,
            'KataGo',
        )
        
        # Set up KataGo
        katago_controller.do_command('boardsize', [str(board_size)])
        katago_controller.do_command('komi', [str(komi)])
        katago_controller.do_command('clear_board')
        
        # Play game
        game_over = False
        consecutive_passes = 0
        winner = None
        
        while not game_over:
            # Determine whose turn it is
            move_number = len(datago_player.move_history)
            current_color = "b" if move_number % 2 == 0 else "w"
            
            # Generate move
            if current_color == datago_color:
                # DataGo's turn
                move = datago_player.genmove()
            else:
                # KataGo's turn
                response = katago_controller.do_command('genmove', [current_color])
                move = response.lower()
                datago_player.handle_move(current_color, move)
            
            # Check for passes
            if move == "pass":
                consecutive_passes += 1
            else:
                consecutive_passes = 0
            
            # Check for resignation or two consecutive passes
            if move == "resign":
                game_over = True
                winner = "w" if current_color == "b" else "b"
                logger.info(f"{current_color} resigned. {winner} wins!")
            elif consecutive_passes >= 2:
                game_over = True
                # Score the game
                score_result = katago_controller.do_command('final_score')
                logger.info(f"Game ended by two passes. Score: {score_result}")
                winner = score_result[0].lower()  # "B" or "W"
            
            # Print board periodically
            if move_number % 20 == 0:
                logger.info(f"\nBoard after move {move_number}:\n" + 
                           datago_player.get_board_string())
        
        # Get final statistics
        stats = datago_player.get_statistics()
        
        result = {
            "winner": winner,
            "datago_color": datago_color,
            "datago_won": (winner == datago_color),
            "total_moves": len(datago_player.move_history),
            "board_size": board_size,
            "komi": komi,
            "statistics": stats,
        }
        
        logger.info(f"Game finished. Winner: {winner}")
        logger.info(f"DataGo statistics: {stats}")
        
        # Save SGF if requested
        if save_sgf:
            # TODO: Implement SGF saving
            logger.info(f"SGF saving not yet implemented. Would save to: {save_sgf}")
        
        return result
    
    finally:
        # Clean up
        if 'katago_controller' in locals():
            katago_controller.close()
        datago_bot.shutdown()


def main():
    """Command-line interface for playing games."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Play Go games with DataGo bot through gomill"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to DataGo bot config file"
    )
    parser.add_argument(
        "--katago-executable",
        required=True,
        help="Path to KataGo executable"
    )
    parser.add_argument(
        "--katago-model",
        required=True,
        help="Path to KataGo model file"
    )
    parser.add_argument(
        "--katago-config",
        required=True,
        help="Path to KataGo config file"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=19,
        help="Board size (default: 19)"
    )
    parser.add_argument(
        "--komi",
        type=float,
        default=7.5,
        help="Komi value (default: 7.5)"
    )
    parser.add_argument(
        "--color",
        choices=["b", "w", "black", "white"],
        default="b",
        help="DataGo bot color (default: black)"
    )
    parser.add_argument(
        "--save-sgf",
        help="Path to save SGF file"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--output-json",
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    # Normalize color
    color = args.color[0].lower()  # "b" or "w"
    
    # Play games
    all_results = []
    for game_num in range(args.num_games):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting game {game_num + 1}/{args.num_games}")
        logger.info(f"{'='*60}\n")
        
        result = play_game_vs_katago(
            datago_config_path=args.config,
            katago_executable=args.katago_executable,
            katago_model=args.katago_model,
            katago_config=args.katago_config,
            board_size=args.board_size,
            komi=args.komi,
            datago_color=color,
            save_sgf=args.save_sgf if args.num_games == 1 else 
                     f"{args.save_sgf}.{game_num}.sgf" if args.save_sgf else None,
        )
        
        all_results.append(result)
        
        # Alternate colors if playing multiple games
        if args.num_games > 1:
            color = "w" if color == "b" else "b"
    
    # Summary
    if args.num_games > 1:
        wins = sum(1 for r in all_results if r['datago_won'])
        win_rate = wins / args.num_games
        logger.info(f"\n{'='*60}")
        logger.info(f"Results: {wins}/{args.num_games} wins ({win_rate:.1%})")
        logger.info(f"{'='*60}\n")
    
    # Save results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
