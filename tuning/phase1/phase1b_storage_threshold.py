"""
Phase 1b: Storage Threshold Tuning
Determines optimal storage threshold after Phase 1 uncertainty weights are fixed.

Hardware: NVIDIA A100
Time Budget: 8-10 hours
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


@dataclass
class StorageThresholdConfig:
    """Configuration for storage threshold testing"""
    percentile: float  # Store top X% of uncertain positions
    absolute_threshold: float  # Computed from percentile
    config_id: str = ""
    
    def __post_init__(self):
        if not self.config_id:
            self.config_id = f"percentile_{self.percentile:.1f}_threshold_{self.absolute_threshold:.3f}"


@dataclass
class StorageMetrics:
    """Metrics for storage threshold evaluation"""
    config_id: str
    win_rate: float
    total_games: int
    database_size_mb: float
    avg_positions_per_game: float
    retrieval_time_ms: float
    storage_time_ms: float
    cache_hit_rate: float  # % of queries that found a match


class StorageThresholdTuner:
    """Tunes the storage threshold for RAG database"""
    
    def __init__(self,
                 phase1_config_path: str,
                 output_dir: str = "./tuning_results/phase1b",
                 num_games_per_threshold: int = 100,
                 max_database_size_gb: int = 20):
        """
        Args:
            phase1_config_path: Path to best config from Phase 1
            output_dir: Directory to save results
            num_games_per_threshold: Number of games to test per threshold
            max_database_size_gb: Maximum allowed database size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Phase 1 best config
        with open(phase1_config_path, 'r') as f:
            phase1_data = json.load(f)
            self.phase1_config = phase1_data['config']
        
        print(f"Loaded Phase 1 config: w1={self.phase1_config['w1']:.3f}, w2={self.phase1_config['w2']:.3f}")
        
        self.num_games_per_threshold = num_games_per_threshold
        self.max_database_size_gb = max_database_size_gb
        self.results: List[Dict] = []
    
    def estimate_percentiles_from_sample(self, num_sample_games: int = 500) -> Dict[float, float]:
        """
        Run sample games to estimate uncertainty score distribution.
        This helps map percentiles to absolute threshold values.
        
        Args:
            num_sample_games: Number of games to sample
            
        Returns:
            Dictionary mapping percentile to absolute threshold value
        """
        print(f"\nRunning {num_sample_games} sample games to estimate percentiles...")
        
        # PLACEHOLDER: Replace with actual game execution
        # This should:
        # 1. Run games and collect all uncertainty scores
        # 2. Build distribution of uncertainty scores
        # 3. Map percentiles to absolute thresholds
        
        # Simulated uncertainty scores for demonstration
        all_uncertainty_scores = []
        for _ in range(num_sample_games):
            # Simulate a game with ~250 positions
            game_scores = np.random.beta(2, 5, size=250)  # Skewed distribution
            all_uncertainty_scores.extend(game_scores)
        
        all_uncertainty_scores = np.array(all_uncertainty_scores)
        
        # Calculate percentile thresholds
        percentiles_to_test = [5, 10, 15, 20]
        percentile_map = {}
        
        for percentile in percentiles_to_test:
            # Top X% means we want the (100-X) percentile threshold
            threshold = np.percentile(all_uncertainty_scores, 100 - percentile)
            percentile_map[percentile] = threshold
            print(f"  Top {percentile}% → threshold {threshold:.4f}")
        
        # Save percentile mapping
        mapping_file = self.output_dir / "percentile_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(percentile_map, f, indent=2)
        
        # Plot distribution
        self.plot_uncertainty_distribution(all_uncertainty_scores, percentile_map)
        
        return percentile_map
    
    def plot_uncertainty_distribution(self, scores: np.ndarray, percentile_map: Dict[float, float]):
        """Plot uncertainty score distribution with threshold lines"""
        plt.figure(figsize=(12, 6))
        
        plt.hist(scores, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Uncertainty Score')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Score Distribution')
        
        # Add vertical lines for thresholds
        colors = ['red', 'orange', 'yellow', 'green']
        for (percentile, threshold), color in zip(sorted(percentile_map.items()), colors):
            plt.axvline(threshold, color=color, linestyle='--', linewidth=2,
                       label=f'Top {percentile}% (threshold={threshold:.3f})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = self.output_dir / "uncertainty_distribution.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved distribution plot to {plot_file}")
        plt.close()
    
    def evaluate_threshold(self, config: StorageThresholdConfig) -> StorageMetrics:
        """
        Evaluate a specific storage threshold.
        
        Args:
            config: Storage threshold configuration
            
        Returns:
            Storage metrics for this threshold
        """
        print(f"\nEvaluating threshold: Top {config.percentile}% (abs={config.absolute_threshold:.4f})")
        
        # PLACEHOLDER: Replace with actual game execution and monitoring
        # This should:
        # 1. Run games with this storage threshold
        # 2. Monitor database growth
        # 3. Track retrieval performance
        # 4. Measure win rate
        
        # Simulated results for demonstration
        import random
        
        # Simulate database growth based on percentile
        positions_per_game = 250 * (config.percentile / 100)
        database_size_mb = positions_per_game * self.num_games_per_threshold * 0.05  # 50KB per position
        
        # Simulate metrics
        metrics = StorageMetrics(
            config_id=config.config_id,
            win_rate=random.uniform(0.48, 0.55),  # Replace with actual win rate
            total_games=self.num_games_per_threshold,
            database_size_mb=database_size_mb,
            avg_positions_per_game=positions_per_game,
            retrieval_time_ms=random.uniform(1, 5),
            storage_time_ms=random.uniform(2, 8),
            cache_hit_rate=random.uniform(0.1, 0.3)
        )
        
        print(f"  Win rate: {metrics.win_rate:.3f}")
        print(f"  Database size: {metrics.database_size_mb:.1f} MB")
        print(f"  Avg positions stored per game: {metrics.avg_positions_per_game:.1f}")
        print(f"  Retrieval time: {metrics.retrieval_time_ms:.2f} ms")
        print(f"  Cache hit rate: {metrics.cache_hit_rate:.3f}")
        
        # Check if database size exceeds limit
        if database_size_mb / 1024 > self.max_database_size_gb:
            print(f"  WARNING: Database would exceed {self.max_database_size_gb}GB limit!")
        
        return metrics
    
    def run_tuning(self) -> Tuple[StorageThresholdConfig, StorageMetrics]:
        """Run storage threshold tuning"""
        print("="*80)
        print("PHASE 1B: STORAGE THRESHOLD TUNING")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Games per threshold: {self.num_games_per_threshold}")
        print(f"Max database size: {self.max_database_size_gb} GB")
        print("="*80)
        
        # Step 1: Estimate percentile thresholds
        percentile_map = self.estimate_percentiles_from_sample()
        
        # Step 2: Generate configs to test
        configs = []
        for percentile, threshold in percentile_map.items():
            config = StorageThresholdConfig(
                percentile=percentile,
                absolute_threshold=threshold
            )
            configs.append(config)
        
        # Step 3: Evaluate each threshold
        results = []
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing threshold...")
            metrics = self.evaluate_threshold(config)
            
            result = {
                "config": asdict(config),
                "metrics": asdict(metrics)
            }
            results.append(result)
            
            # Save intermediate results
            result_file = self.output_dir / f"{config.config_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Step 4: Analyze results and find optimal threshold
        best_config, best_metrics = self.find_optimal_threshold(configs, results)
        
        # Step 5: Save all results
        summary_file = self.output_dir / "storage_threshold_results.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "phase1_config": self.phase1_config,
                "percentile_mapping": percentile_map,
                "all_results": results,
                "best_config": asdict(best_config),
                "best_metrics": asdict(best_metrics)
            }, f, indent=2)
        
        # Step 6: Create visualization
        self.plot_threshold_comparison(configs, results)
        
        print("\n" + "="*80)
        print("OPTIMAL STORAGE THRESHOLD")
        print("="*80)
        print(f"Percentile: Top {best_config.percentile}%")
        print(f"Absolute threshold: {best_config.absolute_threshold:.4f}")
        print(f"Win rate: {best_metrics.win_rate:.3f}")
        print(f"Database size: {best_metrics.database_size_mb:.1f} MB")
        print(f"Positions per game: {best_metrics.avg_positions_per_game:.1f}")
        print("="*80)
        
        return best_config, best_metrics
    
    def find_optimal_threshold(self, configs: List[StorageThresholdConfig], 
                               results: List[Dict]) -> Tuple[StorageThresholdConfig, StorageMetrics]:
        """
        Find optimal threshold balancing win rate and database size.
        
        Strategy: Maximize win rate while keeping database size reasonable
        """
        # Extract metrics
        metrics_list = [StorageMetrics(**r['metrics']) for r in results]
        
        # Compute score: win_rate - penalty for large database
        scores = []
        for metrics in metrics_list:
            # Penalize if database grows too large
            size_penalty = 0
            if metrics.database_size_mb / 1024 > self.max_database_size_gb * 0.8:
                size_penalty = 0.05  # 5% penalty if near limit
            if metrics.database_size_mb / 1024 > self.max_database_size_gb:
                size_penalty = 0.20  # 20% penalty if exceeding limit
            
            score = metrics.win_rate - size_penalty
            scores.append(score)
        
        # Find best score
        best_idx = np.argmax(scores)
        best_config = configs[best_idx]
        best_metrics = metrics_list[best_idx]
        
        return best_config, best_metrics
    
    def plot_threshold_comparison(self, configs: List[StorageThresholdConfig], results: List[Dict]):
        """Create comparison plots for different thresholds"""
        metrics_list = [StorageMetrics(**r['metrics']) for r in results]
        percentiles = [c.percentile for c in configs]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Win rate vs percentile
        axes[0, 0].plot(percentiles, [m.win_rate for m in metrics_list], 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Storage Percentile (Top X%)')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_title('Win Rate vs Storage Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Database size vs percentile
        axes[0, 1].plot(percentiles, [m.database_size_mb / 1024 for m in metrics_list], 's-', linewidth=2, color='orange')
        axes[0, 1].axhline(self.max_database_size_gb, color='red', linestyle='--', label='Max size limit')
        axes[0, 1].set_xlabel('Storage Percentile (Top X%)')
        axes[0, 1].set_ylabel('Database Size (GB)')
        axes[0, 1].set_title('Database Growth vs Storage Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cache hit rate vs percentile
        axes[1, 0].plot(percentiles, [m.cache_hit_rate for m in metrics_list], '^-', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Storage Percentile (Top X%)')
        axes[1, 0].set_ylabel('Cache Hit Rate')
        axes[1, 0].set_title('Cache Hit Rate vs Storage Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Retrieval time vs percentile
        axes[1, 1].plot(percentiles, [m.retrieval_time_ms for m in metrics_list], 'd-', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Storage Percentile (Top X%)')
        axes[1, 1].set_ylabel('Retrieval Time (ms)')
        axes[1, 1].set_title('Retrieval Performance vs Storage Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "threshold_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison plots to {plot_file}")
        plt.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1b: Storage Threshold Tuning")
    parser.add_argument("--phase1-config", type=str, required=True,
                       help="Path to best config from Phase 1 (best_config_phase1.json)")
    parser.add_argument("--output-dir", type=str, default="./tuning_results/phase1b",
                       help="Output directory for results")
    parser.add_argument("--num-games", type=int, default=100,
                       help="Number of games per threshold")
    parser.add_argument("--max-db-size", type=int, default=20,
                       help="Maximum database size in GB")
    
    args = parser.parse_args()
    
    tuner = StorageThresholdTuner(
        phase1_config_path=args.phase1_config,
        output_dir=args.output_dir,
        num_games_per_threshold=args.num_games,
        max_database_size_gb=args.max_db_size
    )
    
    best_config, best_metrics = tuner.run_tuning()
    
    print("\nPhase 1b tuning completed successfully!")


if __name__ == "__main__":
    main()
