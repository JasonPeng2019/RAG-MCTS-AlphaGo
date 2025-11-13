"""
Phase 1: Uncertainty Detection Parameter Tuning
Tunes w1, w2, and phase_function parameters for RAG-enhanced AlphaGo

Hardware: NVIDIA A100
Time Budget: 18-20 hours
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import concurrent.futures
from datetime import datetime

# Optional: Ray Tune for parallel hyperparameter search
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Using sequential tuning.")

# Optional: Weights & Biases for monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: W&B not available. Using local logging.")


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty detection parameters"""
    w1: float  # Weight for policy cross-entropy
    w2: float  # Weight for value distribution sparseness
    phase_early_multiplier: float  # Early game phase multiplier
    phase_late_multiplier: float  # Late game phase multiplier
    config_id: str = ""
    
    def __post_init__(self):
        if not self.config_id:
            self.config_id = f"w1_{self.w1:.2f}_w2_{self.w2:.2f}_early_{self.phase_early_multiplier:.2f}_late_{self.phase_late_multiplier:.2f}"
    
    def compute_uncertainty(self, policy_entropy: float, value_sparseness: float, 
                          stones_on_board: int, total_stones: int = 361) -> float:
        """
        Compute uncertainty score for a position.
        
        Args:
            policy_entropy: Cross-entropy of policy distribution (E)
            value_sparseness: Sparseness of value distribution (K)
            stones_on_board: Number of stones currently on board
            total_stones: Total possible stones (19x19 = 361)
        
        Returns:
            Uncertainty score
        """
        # Compute phase multiplier (linear interpolation)
        game_progress = stones_on_board / total_stones
        phase_multiplier = (
            self.phase_early_multiplier * (1 - game_progress) + 
            self.phase_late_multiplier * game_progress
        )
        
        # Combined uncertainty score
        uncertainty = (self.w1 * policy_entropy + self.w2 * value_sparseness) * phase_multiplier
        return uncertainty


@dataclass
class GameResult:
    """Result of a single game"""
    config_id: str
    win: bool
    game_length: int
    avg_uncertainty: float
    max_uncertainty: float
    num_stored_positions: int
    computation_time: float
    

class Phase1Tuner:
    """Tunes Phase 1 parameters using parallel game execution"""
    
    def __init__(self, 
                 output_dir: str = "./tuning_results",
                 num_games_per_config: int = 150,
                 parallel_workers: int = 32,
                 early_stopping_games: int = 100,
                 early_stopping_threshold: float = 0.40):
        """
        Args:
            output_dir: Directory to save results
            num_games_per_config: Number of games to run per configuration
            parallel_workers: Number of parallel game workers
            early_stopping_games: Number of games before checking for early stopping
            early_stopping_threshold: Win rate threshold for early stopping (abort if below)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_games_per_config = num_games_per_config
        self.parallel_workers = parallel_workers
        self.early_stopping_games = early_stopping_games
        self.early_stopping_threshold = early_stopping_threshold
        
        self.results: List[Dict] = []
        
        # Initialize W&B if available
        if WANDB_AVAILABLE:
            wandb.init(project="rag-alphago-phase1", name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def generate_configs(self) -> List[UncertaintyConfig]:
        """Generate grid of configurations to test"""
        configs = []
        
        # Reduced grid for 4-day timeline: 3x3 = 9 combinations
        w1_values = [0.3, 0.5, 0.7]
        phase_early_values = [0.9, 1.0, 1.1]
        phase_late_values = [0.9, 1.0, 1.1]
        
        for w1 in w1_values:
            w2 = 1.0 - w1  # Normalized weights
            for early in phase_early_values:
                for late in phase_late_values:
                    config = UncertaintyConfig(
                        w1=w1,
                        w2=w2,
                        phase_early_multiplier=early,
                        phase_late_multiplier=late
                    )
                    configs.append(config)
        
        print(f"Generated {len(configs)} configurations to test")
        return configs
    
    def run_single_game(self, config: UncertaintyConfig, game_id: int) -> GameResult:
        """
        Run a single game with given configuration.
        
        This is a placeholder - you need to integrate with your actual KataGo client.
        Replace this with actual game execution logic.
        """
        # PLACEHOLDER: Replace with actual KataGo game execution
        # This should:
        # 1. Start a game with the RAG system using this config
        # 2. Track uncertainty scores throughout the game
        # 3. Store positions that exceed the uncertainty threshold
        # 4. Return game results
        
        # Simulated game for demonstration
        import random
        time.sleep(0.1)  # Simulate game time
        
        result = GameResult(
            config_id=config.config_id,
            win=random.random() > 0.5,  # Replace with actual win/loss
            game_length=random.randint(150, 300),
            avg_uncertainty=random.uniform(0.3, 0.8),
            max_uncertainty=random.uniform(0.7, 1.5),
            num_stored_positions=random.randint(10, 50),
            computation_time=random.uniform(30, 120)
        )
        
        return result
    
    def evaluate_config(self, config: UncertaintyConfig) -> Dict:
        """
        Evaluate a configuration by running multiple games in parallel.
        
        Returns aggregated results for this configuration.
        """
        print(f"\nEvaluating config: {config.config_id}")
        start_time = time.time()
        
        results = []
        
        # Run games in parallel batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Check for early stopping
            for batch_start in range(0, self.num_games_per_config, self.early_stopping_games):
                batch_size = min(self.early_stopping_games, self.num_games_per_config - batch_start)
                
                # Submit batch of games
                futures = [
                    executor.submit(self.run_single_game, config, batch_start + i)
                    for i in range(batch_size)
                ]
                
                # Collect results
                batch_results = [future.result() for future in concurrent.futures.as_completed(futures)]
                results.extend(batch_results)
                
                # Early stopping check
                if len(results) >= self.early_stopping_games:
                    win_rate = sum(r.win for r in results) / len(results)
                    print(f"  After {len(results)} games: Win rate = {win_rate:.3f}")
                    
                    if win_rate < self.early_stopping_threshold:
                        print(f"  Early stopping triggered (win rate {win_rate:.3f} < {self.early_stopping_threshold:.3f})")
                        break
                    
                    # Log to W&B
                    if WANDB_AVAILABLE:
                        wandb.log({
                            f"win_rate_{config.config_id}": win_rate,
                            "games_completed": len(results)
                        })
        
        # Aggregate results
        total_games = len(results)
        wins = sum(r.win for r in results)
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        aggregated = {
            "config": asdict(config),
            "total_games": total_games,
            "wins": wins,
            "win_rate": win_rate,
            "avg_game_length": np.mean([r.game_length for r in results]),
            "avg_uncertainty": np.mean([r.avg_uncertainty for r in results]),
            "avg_stored_positions": np.mean([r.num_stored_positions for r in results]),
            "total_computation_time": time.time() - start_time,
            "avg_time_per_game": np.mean([r.computation_time for r in results])
        }
        
        print(f"  Completed {total_games} games in {aggregated['total_computation_time']:.1f}s")
        print(f"  Win rate: {win_rate:.3f} ({wins}/{total_games})")
        
        # Save intermediate results
        self.save_result(aggregated)
        
        return aggregated
    
    def save_result(self, result: Dict):
        """Save a single configuration result"""
        self.results.append(result)
        
        # Save to JSON
        result_file = self.output_dir / "phase1_results.json"
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save individual config result
        config_file = self.output_dir / f"{result['config']['config_id']}.json"
        with open(config_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def find_best_config(self) -> Tuple[UncertaintyConfig, Dict]:
        """Find the best performing configuration"""
        if not self.results:
            raise ValueError("No results available")
        
        # Sort by win rate
        sorted_results = sorted(self.results, key=lambda x: x['win_rate'], reverse=True)
        best = sorted_results[0]
        
        best_config = UncertaintyConfig(**best['config'])
        
        print("\n" + "="*80)
        print("BEST CONFIGURATION")
        print("="*80)
        print(f"Config ID: {best_config.config_id}")
        print(f"w1 (policy entropy weight): {best_config.w1:.3f}")
        print(f"w2 (value sparseness weight): {best_config.w2:.3f}")
        print(f"Phase early multiplier: {best_config.phase_early_multiplier:.3f}")
        print(f"Phase late multiplier: {best_config.phase_late_multiplier:.3f}")
        print(f"\nWin rate: {best['win_rate']:.3f} ({best['wins']}/{best['total_games']} games)")
        print(f"Avg game length: {best['avg_game_length']:.1f}")
        print(f"Avg stored positions: {best['avg_stored_positions']:.1f}")
        print("="*80)
        
        return best_config, best
    
    def run_tuning(self):
        """Run full Phase 1 tuning"""
        print("="*80)
        print("PHASE 1: UNCERTAINTY DETECTION PARAMETER TUNING")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Games per config: {self.num_games_per_config}")
        print(f"Parallel workers: {self.parallel_workers}")
        print(f"Early stopping: {self.early_stopping_games} games @ {self.early_stopping_threshold:.2f} win rate")
        print("="*80)
        
        configs = self.generate_configs()
        
        start_time = time.time()
        
        # Evaluate each configuration
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing configuration...")
            result = self.evaluate_config(config)
            
            # Estimate time remaining
            elapsed = time.time() - start_time
            avg_time_per_config = elapsed / i
            remaining_configs = len(configs) - i
            estimated_remaining = avg_time_per_config * remaining_configs
            
            print(f"\nProgress: {i}/{len(configs)} configs completed")
            print(f"Elapsed time: {elapsed/3600:.2f} hours")
            print(f"Estimated remaining: {estimated_remaining/3600:.2f} hours")
        
        # Find and save best configuration
        best_config, best_result = self.find_best_config()
        
        # Save best config separately
        best_config_file = self.output_dir / "best_config_phase1.json"
        with open(best_config_file, 'w') as f:
            json.dump({
                "config": asdict(best_config),
                "results": best_result
            }, f, indent=2)
        
        print(f"\nBest configuration saved to: {best_config_file}")
        
        total_time = time.time() - start_time
        print(f"\nTotal tuning time: {total_time/3600:.2f} hours")
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        return best_config, best_result


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1: Uncertainty Detection Tuning")
    parser.add_argument("--output-dir", type=str, default="./tuning_results/phase1",
                       help="Output directory for results")
    parser.add_argument("--num-games", type=int, default=150,
                       help="Number of games per configuration")
    parser.add_argument("--parallel-workers", type=int, default=32,
                       help="Number of parallel game workers")
    parser.add_argument("--early-stopping-games", type=int, default=100,
                       help="Check for early stopping after this many games")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.40,
                       help="Early stopping win rate threshold")
    
    args = parser.parse_args()
    
    tuner = Phase1Tuner(
        output_dir=args.output_dir,
        num_games_per_config=args.num_games,
        parallel_workers=args.parallel_workers,
        early_stopping_games=args.early_stopping_games,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    best_config, best_result = tuner.run_tuning()
    
    print("\nPhase 1 tuning completed successfully!")
    print(f"Best config ID: {best_config.config_id}")
    print(f"Win rate: {best_result['win_rate']:.3f}")


if __name__ == "__main__":
    main()
