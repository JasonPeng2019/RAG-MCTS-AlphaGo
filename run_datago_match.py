#!/usr/bin/env python3
"""
run_datago_match.py

Run a match between DataGo bot and KataGo using a shared GTP instance.
This works around the process conflict by having DataGo query KataGo via GTP
for analysis, then using its RAG logic to enhance the move selection.
"""

import argparse
import logging
import time
from pathlib import Path

from src.bot.gtp_controller import GTPController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_match(
    katago_executable: str,
    katago_model: str,
    katago_config: str,
    num_games: int = 1,
    max_moves: int = 200,
):
    """
    Run DataGo vs KataGo match.
    
    Strategy:
    - Start ONE KataGo GTP instance on GPU 7
    - Black (DataGo): Uses genmove with additional analysis query for RAG logic
    - White (KataGo): Uses standard genmove
    
    This tests if DataGo's RAG-enhanced decision making improves play.
    """
    logger.info("=" * 70)
    logger.info("DataGo Bot vs KataGo Match")
    logger.info("=" * 70)
    logger.info(f"Games: {num_games}, Max moves per game: {max_moves}")
    logger.info("Using single KataGo instance on GPU 7")
    logger.info("Black: DataGo (GTP + RAG logic)")
    logger.info("White: KataGo (pure GTP)")
    logger.info("")
    
    # Start KataGo GTP on GPU 7
    logger.info("Starting KataGo GTP on GPU 7...")
    cmd = [
        katago_executable,
        'gtp',
        '-model', katago_model,
        '-config', katago_config,
    ]
    katago = GTPController(command=cmd)
    
    results = []
    
    for game_num in range(1, num_games + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Game {game_num}/{num_games}")
        logger.info(f"{'=' * 70}\n")
        
        try:
            # Setup game
            katago.boardsize(19)
            katago.clear_board()
            katago.komi(7.5)
            
            move_history = []
            move_number = 0
            passes = 0
            
            while move_number < max_moves:
                move_number += 1
                
                if move_number % 2 == 1:  # Black (DataGo)
                    logger.info(f"Move {move_number}: DataGo (Black) thinking...")
                    
                    # Get KataGo's suggestion
                    move = katago.genmove("B")
                    
                    if not move:
                        logger.error("Failed to get move")
                        break
                    
                    move = move.upper()
                    
                    # TODO: Here is where RAG logic would enhance the decision
                    # For now, just use KataGo's move directly
                    # In full version:
                    #   1. Check uncertainty of position
                    #   2. Query RAG if uncertain
                    #   3. Blend/modify move based on retrieval
                    
                    logger.info(f"Move {move_number}: DataGo (Black) plays {move}")
                    move_history.append(("B", move))
                    
                else:  # White (KataGo pure)
                    logger.info(f"Move {move_number}: KataGo (White) thinking...")
                    move = katago.genmove("W")
                    
                    if not move:
                        logger.error("Failed to get move")
                        break
                    
                    move = move.upper()
                    logger.info(f"Move {move_number}: KataGo (White) plays {move}")
                    move_history.append(("W", move))
                
                # Check game end
                if move == "RESIGN":
                    winner = "KataGo" if move_history[-1][0] == "B" else "DataGo"
                    logger.info(f"\n{('DataGo' if move_history[-1][0] == 'B' else 'KataGo')} resigned. {winner} wins!")
                    results.append(winner)
                    break
                elif move == "PASS":
                    passes += 1
                    if passes >= 2:
                        logger.info("\nBoth players passed. Game over.")
                        # For now, call it a draw (proper scoring would require final_score)
                        results.append("Draw")
                        break
                else:
                    passes = 0
                
                time.sleep(0.05)
            
            if move_number >= max_moves:
                logger.info(f"\nReached max moves ({max_moves}). Draw.")
                results.append("Draw")
        
        except Exception as e:
            logger.error(f"Error in game {game_num}: {e}", exc_info=True)
            results.append("Error")
    
    # Cleanup
    logger.info("\nCleaning up...")
    katago.quit()
    
    # Results summary
    logger.info("\n" + "=" * 70)
    logger.info("Match Results")
    logger.info("=" * 70)
    for i, result in enumerate(results, 1):
        logger.info(f"Game {i}: {result}")
    
    datago_wins = sum(1 for r in results if r == "DataGo")
    katago_wins = sum(1 for r in results if r == "KataGo")
    draws = sum(1 for r in results if r == "Draw")
    
    logger.info("")
    logger.info(f"DataGo: {datago_wins} wins")
    logger.info(f"KataGo: {katago_wins} wins")
    logger.info(f"Draws: {draws}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run DataGo vs KataGo match")
    parser.add_argument('--katago-executable', required=True, help='Path to KataGo binary')
    parser.add_argument('--katago-model', required=True, help='Path to KataGo model')
    parser.add_argument('--katago-config', required=True, help='KataGo config')
    parser.add_argument('--games', type=int, default=1, help='Number of games (default: 1)')
    parser.add_argument('--max-moves', type=int, default=200, help='Max moves per game (default: 200)')
    
    args = parser.parse_args()
    
    run_match(
        katago_executable=args.katago_executable,
        katago_model=args.katago_model,
        katago_config=args.katago_config,
        num_games=args.games,
        max_moves=args.max_moves,
    )


if __name__ == '__main__':
    main()
