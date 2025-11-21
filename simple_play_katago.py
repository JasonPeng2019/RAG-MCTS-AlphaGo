#!/usr/bin/env python3
"""
simple_play_katago.py

Simplified script to test KataGo vs KataGo on GPU 7.
This avoids the complex DataGo bot for initial testing.
"""

import sys
import logging
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from src.bot.gtp_controller import GTPController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_defaults():
    """Resolve default KataGo paths relative to repository root."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    katago_repo = repo_root / "katago_repo"
    return {
        "katago_exe": katago_repo / "KataGo" / "cpp" / "build-opencl" / "katago",
        "model_path": katago_repo / "run" / "default_model.bin.gz",
        "config_path": katago_repo / "run" / "gtp_800visits.cfg",
    }


def play_simple_game(katago_exe: Path, model_path: Path, config_path: Path):
    """Play a simple self-play game with two KataGo instances."""
    
    logger.info("=" * 70)
    logger.info("KataGo Self-Play Test on GPU 7")
    logger.info("=" * 70)
    
    # Start two KataGo instances
    logger.info("Starting KataGo Black...")
    cmd = [str(katago_exe), 'gtp', '-model', str(model_path), '-config', str(config_path)]
    black = GTPController(cmd)
    
    logger.info("Starting KataGo White...")
    white = GTPController(cmd)
    
    # Setup game
    board_size = 19
    komi = 7.5
    
    for player in [black, white]:
        player.boardsize(board_size)
        player.clear_board()
        player.komi(komi)
    
    logger.info(f"\nGame settings: {board_size}x{board_size}, komi={komi}")
    logger.info("Starting game...\n")
    
    # Play game
    move_count = 0
    max_moves = 50  # Limit for testing
    
    while move_count < max_moves:
        move_count += 1
        
        # Black's turn
        logger.info(f"Move {move_count}: Black thinking...")
        black_move = black.genmove('black')
        if not black_move or black_move.lower() in ['resign', 'pass']:
            logger.info(f"Black {black_move}")
            break
        
        logger.info(f"Move {move_count}: Black plays {black_move}")
        white.play('black', black_move)
        
        # White's turn
        move_count += 1
        logger.info(f"Move {move_count}: White thinking...")
        white_move = white.genmove('white')
        if not white_move or white_move.lower() in ['resign', 'pass']:
            logger.info(f"White {white_move}")
            break
        
        logger.info(f"Move {move_count}: White plays {white_move}")
        black.play('white', white_move)
    
    logger.info(f"\nGame finished after {move_count} moves")
    
    # Cleanup
    black.quit()
    white.quit()
    
    logger.info("\n" + "=" * 70)
    logger.info("Test completed successfully!")
    logger.info("=" * 70)


if __name__ == '__main__':
    defaults = resolve_defaults()
    parser = argparse.ArgumentParser(description="Simple KataGo self-play test.")
    parser.add_argument("--katago-executable", type=Path, default=defaults["katago_exe"])
    parser.add_argument("--katago-model", type=Path, default=defaults["model_path"])
    parser.add_argument("--katago-config", type=Path, default=defaults["config_path"])
    args = parser.parse_args()

    try:
        play_simple_game(args.katago_executable, args.katago_model, args.katago_config)
    except KeyboardInterrupt:
        logger.info("\nGame interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
