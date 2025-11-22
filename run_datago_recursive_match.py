#!/usr/bin/env python3
"""
run_datago_recursive_match.py

DataGo bot with recursive deep search and RAG integration.

Algorithm:
1. During MCTS, detect high-uncertainty positions
2. For complex positions:
   a. Query RAG database by sym_hash
   b. If exact match found: Use stored analysis
   c. If no match: Run deep augmented search
   d. Store results in RAG (multiple contexts per sym_hash)
3. During deep search, recursively apply same logic to child positions
4. Respect max_recursion_depth limit
"""

import argparse
import logging
import time
import math
import json
import hashlib
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.bot.gtp_controller import GTPController
from src.memory.index import ANNIndex
from src.memory.schema import MemoryEntry
from src.gating.gate import normalized_entropy, entropy_of_policy
from src.blend.blend import rerank_neighbors, build_retrieval_prior, blend_priors
from src.utils import symmetry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BoardState:
    """Represents a Go board position with symmetry canonicalization."""
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)  # 0=empty, 1=black, -1=white
        self.move_history = []
    
    def copy(self) -> 'BoardState':
        """Create a deep copy of the board state."""
        new_state = BoardState(self.board_size)
        new_state.board = self.board.copy()
        new_state.move_history = self.move_history.copy()
        return new_state
    
    def play_move(self, move: str, color: int):
        """
        Play a move on the board.
        
        Args:
            move: Move in GTP format (e.g., "D4", "pass")
            color: 1 for black, -1 for white
        """
        if move.upper() in ["PASS", "RESIGN"]:
            self.move_history.append((move, color))
            return
        
        # Parse move coordinates (e.g., "D4" -> (3, 3))
        col_letter = move[0].upper()
        row_str = move[1:]
        
        # Convert column letter to index (A=0, B=1, ..., skipping I)
        col = ord(col_letter) - ord('A')
        if col >= 8:  # Skip 'I'
            col -= 1
        
        # Convert row (1-indexed from bottom)
        row = self.board_size - int(row_str)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            self.board[row, col] = color
            self.move_history.append((move, color))
    
    def get_canonical_board(self) -> Tuple[np.ndarray, int]:
        """
        Get canonical representation considering all 8 symmetries.
        
        Returns:
            Tuple of (canonical_board, symmetry_index)
        """
        best_board = None
        best_sym = 0
        
        # Try all 8 symmetries (or 4 if non-square)
        for sym in range(8):
            transformed = np.zeros_like(self.board)
            for row in range(self.board_size):
                for col in range(self.board_size):
                    nx, ny = symmetry.apply_symmetry_xy(col, row, self.board_size, self.board_size, sym)
                    transformed[ny, nx] = self.board[row, col]
            
            # Use lexicographic ordering to pick canonical form
            flat = transformed.flatten()
            if best_board is None or tuple(flat) < tuple(best_board.flatten()):
                best_board = transformed
                best_sym = sym
        
        return best_board, best_sym
    
    def get_sym_hash(self) -> str:
        """
        Create symmetry hash using canonical board representation.
        
        Returns:
            Hex string hash of canonical board
        """
        canonical_board, _ = self.get_canonical_board()
        board_bytes = canonical_board.tobytes()
        hash_obj = hashlib.sha256(board_bytes)
        return hash_obj.hexdigest()[:16]  # Use first 16 chars
    
    def to_string(self) -> str:
        """Convert board to string representation for debugging."""
        chars = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for row in self.board:
            lines.append(''.join(chars[cell] for cell in row))
        return '\n'.join(lines)


@dataclass
class PositionContext:
    """Multiple game contexts for the same position (sym_hash)."""
    sym_hash: str
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_context(self, move: str, uncertainty: float, deep_visits: int,
                   policy: np.ndarray, winrate: float, score_lead: float,
                   game_phase: int, metadata: Dict[str, Any]):
        """Add a new game context for this position."""
        self.contexts.append({
            'move': move,
            'uncertainty': uncertainty,
            'deep_visits': deep_visits,
            'policy': policy.copy(),
            'winrate': winrate,
            'score_lead': score_lead,
            'game_phase': game_phase,
            'metadata': metadata,
            'raw_entry': metadata.get('raw_entry'),
            'timestamp': time.time(),
        })
    
    def get_best_context(
        self,
        current_phase: int,
        current_policy: Optional[np.ndarray],
        current_komi: float,
        current_score_lead: float,
        visits_normalizer: float,
        visits_weight: float = 0.3,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Score cached contexts against the current position.
        
        Returns:
            Tuple of (best_context_or_None, best_score)
        """
        if not self.contexts:
            return None, float("-inf")
        
        best_context = None
        best_score = float("-inf")
        current_phase_norm = current_phase / 361.0
        phase_weight = 0.5
        policy_weight = 0.4
        komi_penalty_weight = 0.05
        score_penalty_weight = 0.05
        policy_norm = None
        if current_policy is not None:
            policy_norm = np.linalg.norm(current_policy) + 1e-9
        
        for ctx in self.contexts:
            # Phase cosine-like similarity (already normalized during storage)
            ctx_phase_norm = ctx['game_phase'] / 361.0
            phase_cos = 1.0 - abs(ctx_phase_norm - current_phase_norm)
            phase_cos = max(phase_cos, -1.0)
            
            # Policy cosine similarity
            if current_policy is None:
                policy_cos = 0.0
            else:
                ctx_policy = ctx['policy']
                ctx_norm = np.linalg.norm(ctx_policy) + 1e-9
                policy_cos = float(np.dot(ctx_policy, current_policy) / (policy_norm * ctx_norm))
            
            ctx_meta = ctx['metadata'] or {}
            raw_entry = ctx_meta.get('raw_entry') or {}
            ctx_komi = raw_entry.get('komi', ctx_meta.get('komi', current_komi))
            visits_norm = max(visits_normalizer, 1.0)
            visits_score = min(ctx['deep_visits'] / visits_norm, 1.0)
            score = (
                phase_weight * phase_cos +
                policy_weight * policy_cos +
                visits_weight * visits_score -
                komi_penalty_weight * abs(ctx_komi - current_komi) -
                score_penalty_weight * abs(ctx['score_lead'] - current_score_lead)
            )
            
            logger.debug(
                "Context scoring sym_hash=%s move=%s phase_cos=%.3f policy_cos=%.3f visits_score=%.3f komi_delta=%.2f score_delta=%.2f total=%.3f",
                ctx['metadata'].get('source', 'cache'),
                ctx['move'],
                phase_cos,
                policy_cos,
                visits_score,
                ctx_komi - current_komi,
                ctx['score_lead'] - current_score_lead,
                score,
            )

            if score > best_score:
                best_score = score
                best_context = ctx
        
        return best_context, best_score


class RecursiveDataGoPlayer:
    """DataGo player with recursive deep search and RAG integration."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        katago: GTPController,
        force_rag_sym: Optional[str] = None,
    ):
        self.config = config
        self.katago = katago
        self.force_rag_sym = force_rag_sym.lower() if force_rag_sym else None
        
        # Initialize RAG index
        logger.info("Initializing RAG database with multi-context support...")
        rag_config = config['rag_database']
        embedding_dim = rag_config.get('embedding_dim', 64)
        
        self.rag_index = ANNIndex(
            dim=embedding_dim,
            space=rag_config['ann'].get('distance_metric', 'cosine'),
        )
        self.rag_data_path = Path(rag_config['database_path'])
        self.added_entries_file = None
        added_base = None
        if self.rag_data_path.exists() and self.rag_data_path.is_dir():
            added_base = self.rag_data_path
        else:
            # Use parent directory if database path points to a file/non-existent path
            added_base = self.rag_data_path.parent if self.rag_data_path.parent != Path('') else Path('.')
        self.added_entries_file = added_base / "added_entries.json"
        
        # Position database: sym_hash -> PositionContext
        self.position_db: Dict[str, PositionContext] = {}
        
        # Recursion tracking
        self.current_recursion_depth = 0
        self.max_recursion_depth = config['recursion']['max_recursion_depth']
        
        # Thresholds
        self.uncertainty_threshold = config['rag_query']['uncertainty_threshold']
        deep_search_offset = config['rag_query'].get('deep_search_offset', 0.0)
        self.deep_search_threshold = self.uncertainty_threshold + deep_search_offset
        
        # Deep MCTS settings
        self.deep_visits = config['deep_mcts']['max_visits']
        self.standard_visits = config['katago']['visits']
        self.visits_normalizer = max(1, self.deep_visits)
        self.min_rag_relevance = config['rag_query'].get('min_relevance', 0.5)
        
        # Statistics
        self.stats = {
            'moves': 0,
            'rag_queries': 0,
            'rag_hits': 0,
            'exact_matches': 0,
            'deep_searches': 0,
            'recursive_searches': 0,
            'positions_stored': 0,
            'contexts_added': 0,
            'total_uncertainty': 0.0,
        }
        
        # Board state tracking
        self.board_state = BoardState(config['katago']['board_size'])
        self.move_history = []
        self.stones_on_board = 0

        # Pre-load any offline RAG data (legacy database or selfplay output)
        db_path = Path(rag_config['database_path'])
        self._load_initial_rag_data(db_path)
        
        logger.info(f"Recursion depth: {self.max_recursion_depth}")
        logger.info(f"Deep search threshold: {self.deep_search_threshold:.3f}")
        logger.info(f"Deep search visits: {self.deep_visits}, Standard visits: {self.standard_visits}")
    
    def calculate_uncertainty(self, policy: np.ndarray, move_info: List[Dict]) -> float:
        """Calculate uncertainty score using config parameters."""
        cfg = self.config['uncertainty_detection']
        
        # E: Normalized entropy of policy
        E = normalized_entropy(policy)
        
        # K: Value distribution sparseness
        if len(move_info) > 1:
            visits = np.array([m['visits'] for m in move_info[:10]])
            visit_probs = visits / (visits.sum() + 1e-9)
            K = normalized_entropy(visit_probs)
        else:
            K = 0.0
        
        # Phase adjustment
        phase_type = cfg['phase_function_type']
        coeffs = cfg['phase_function_coefficients']
        stones_ratio = self.stones_on_board / 361.0
        
        if phase_type == 'linear':
            phase = coeffs[0] * stones_ratio + coeffs[1]
        elif phase_type == 'exponential':
            phase = coeffs[0] * math.exp(coeffs[1] * stones_ratio) + coeffs[2]
        else:  # piecewise
            if self.stones_on_board < 120:
                phase = coeffs[0]
            elif self.stones_on_board < 240:
                phase = coeffs[1]
            else:
                phase = coeffs[2]
        
        # Combined uncertainty
        w1, w2 = cfg['w1'], cfg['w2']
        uncertainty = (w1 * E + w2 * K) * phase
        
        return float(uncertainty)
    
    def query_rag_exact(self, sym_hash: str) -> Optional[PositionContext]:
        """Query RAG database for exact sym_hash match."""
        return self.position_db.get(sym_hash)

    def _load_initial_rag_data(self, db_path: Path):
        """Load cached RAG data from file or directory."""
        if not db_path.exists():
            logger.info(f"No RAG data found at {db_path}, starting empty cache")
            return

        try:
            if db_path.is_dir():
                new_positions, contexts = self._load_selfplay_directory(db_path)
            else:
                new_positions, contexts = self._load_selfplay_file(db_path)
        except Exception as exc:
            logger.error(f"Failed loading RAG data from {db_path}: {exc}", exc_info=True)
            return

        if contexts > 0:
            logger.info(
                "Loaded %d cached contexts spanning %d unique positions from %s",
                contexts,
                len(self.position_db),
                db_path,
            )
        else:
            logger.info(f"No usable RAG entries found at {db_path}")

    def _load_selfplay_directory(self, directory: Path) -> Tuple[int, int]:
        """
        Load flagged positions generated by selfplay jobs.

        Args:
            directory: Root directory containing per-game JSON files

        Returns:
            Tuple(new_positions, contexts_added)
        """
        json_files = list(directory.rglob("*.json"))
        if not json_files:
            return 0, 0

        new_positions = 0
        contexts = 0
        for json_path in json_files:
            added_pos, added_ctx = self._load_selfplay_file(json_path)
            new_positions += added_pos
            contexts += added_ctx
        return new_positions, contexts

    def _load_selfplay_file(self, json_path: Path) -> Tuple[int, int]:
        """Load a single selfplay-style JSON file."""
        try:
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    entries = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed line in {json_path}")
                    if not entries:
                        return 0, 0
                    data = {
                        'settings': {
                            'board_size': entries[0].get('history_context', {}).get('board_x_size', 19)
                        },
                        'flagged_positions': entries,
                    }
        except Exception as exc:
            logger.warning(f"Skipping {json_path}: {exc}")
            return 0, 0

        flagged_positions = data.get('flagged_positions')
        if not isinstance(flagged_positions, list):
            logger.warning(f"{json_path} does not contain flagged_positions array")
            return 0, 0

        board_size_hint = data.get('settings', {}).get('board_size')
        new_positions = 0
        contexts = 0
        for entry in flagged_positions:
            added_pos, added_ctx = self._ingest_selfplay_entry(
                entry,
                board_size_hint,
                json_path,
            )
            new_positions += added_pos
            contexts += added_ctx
        return new_positions, contexts

    def _ingest_selfplay_entry(
        self,
        entry: Dict[str, Any],
        board_size_hint: Optional[int],
        source_path: Path,
    ) -> Tuple[int, int]:
        """
        Convert a flagged position record into a cached PositionContext.

        Returns:
            Tuple(new_positions_added, contexts_added)
        """
        sym_hash = entry.get('sym_hash') or entry.get('state_hash')
        if not sym_hash:
            return 0, 0

        entry_board_size = (
            entry.get('history_context', {}).get('board_x_size') or board_size_hint
        )
        if entry_board_size and entry_board_size != self.board_state.board_size:
            # Ignore positions from a different board size (e.g., 7x7 smoke tests)
            return 0, 0

        deep_result = entry.get('deep_result') or {}
        policy_values = deep_result.get('policy') or entry.get('policy')
        if not policy_values:
            return 0, 0

        policy = self._prepare_policy_array(policy_values)
        move_obj = deep_result.get('best_move') or entry.get('best_move') or {}
        move = move_obj.get('move')
        if not move:
            return 0, 0

        winrate = deep_result.get('winrate', entry.get('winrate', 0.5))
        score_lead = deep_result.get('score_lead', entry.get('score_lead', 0.0))
        deep_visits = deep_result.get('visits', 0)
        stone_total = (entry.get('stone_count') or {}).get('total', 0)
        move_number = entry.get('move_number', 0)
        metadata = {
            'source': str(source_path),
            'query_id': entry.get('query_id'),
            'player_to_move': entry.get('player_to_move'),
            'recursion_depth': 0,
            'move_number': move_number,
            'raw_entry': entry,
            'komi': entry.get('komi', self.config['katago'].get('komi', 7.5)),
        }

        uncertainty_metrics = entry.get('uncertainty_metrics') or {}
        uncertainty = uncertainty_metrics.get('combined_score')
        if uncertainty is None:
            # Fall back to simple aggregation if combined score is missing
            entropy = uncertainty_metrics.get('policy_entropy', 0.0)
            value_var = uncertainty_metrics.get('value_variance', 0.0)
            uncertainty = float(entropy + value_var)

        position_added = self._add_cached_context(
            sym_hash=sym_hash,
            move=move,
            policy=policy,
            winrate=winrate,
            score_lead=score_lead,
            deep_visits=deep_visits,
            uncertainty=uncertainty,
            game_phase=stone_total or move_number,
            metadata=metadata,
        )
        return (1 if position_added else 0, 1)

    def _prepare_policy_array(self, values: List[float]) -> np.ndarray:
        """Convert stored policy values into a normalized numpy vector."""
        arr = np.array(values, dtype=np.float32).flatten()
        arr[arr < 0] = 0.0  # Some selfplay dumps mark illegal moves with -1

        board_area = self.board_state.board_size ** 2
        if arr.size < board_area:
            arr = np.pad(arr, (0, board_area - arr.size))
        elif arr.size > board_area:
            arr = arr[:board_area]

        total = float(arr.sum())
        if total > 0:
            arr /= total
        else:
            arr = np.ones(board_area, dtype=np.float32) / board_area
        return arr

    def _add_cached_context(
        self,
        sym_hash: str,
        move: str,
        policy: np.ndarray,
        winrate: float,
        score_lead: float,
        deep_visits: int,
        uncertainty: float,
        game_phase: int,
        metadata: Dict[str, Any],
    ) -> bool:
        """Insert a context into the in-memory cache and ANN index."""
        position_created = False
        if sym_hash not in self.position_db:
            self.position_db[sym_hash] = PositionContext(sym_hash=sym_hash)
            position_created = True

            embedding = policy[:64] if len(policy) >= 64 else np.pad(policy, (0, 64 - len(policy)))
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
            entry = MemoryEntry.create(
                embed=embedding,
                canonical_board=sym_hash,
                best_moves=[{'move': move, 'prob': 1.0}],
                importance=uncertainty,
            )
            self.rag_index.add(entry)

        context = self.position_db[sym_hash]
        context.add_context(
            move=move,
            uncertainty=uncertainty,
            deep_visits=deep_visits,
            policy=policy,
            winrate=winrate,
            score_lead=score_lead,
            game_phase=game_phase or 0,
            metadata=metadata,
        )
        return position_created

    def _build_raw_entry(
        self,
        sym_hash: str,
        move: str,
        analysis: Dict[str, Any],
        board_state: BoardState,
        uncertainty: float,
    ) -> Dict[str, Any]:
        """Generate a selfplay_output-like record for a cached position."""
        board_flat, stone_count = self._serialize_board(board_state)
        policy_arr = analysis.get('policy', np.zeros(board_state.board_size ** 2))
        if isinstance(policy_arr, np.ndarray):
            policy_list = policy_arr.astype(float).tolist()
        else:
            policy_list = list(policy_arr)

        player_to_move = 'B' if len(board_state.move_history) % 2 == 0 else 'W'
        best_move_payload = {
            'symmetry_index': 0,
            'hash': sym_hash,
            'move': move,
        }
        combined_visits = analysis.get('deep_visits', analysis.get('visits', 0))
        deep_result = {
            'available': True,
            'status': 'cached',
            'visits': combined_visits,
            'elapsed_seconds': 0.0,
            'sym_hash': sym_hash,
            'state_hash': sym_hash,
            'player_to_move': player_to_move,
            'symmetry_index': 0,
            'move_number': len(board_state.move_history),
            'komi': self.config['katago'].get('komi', 7.5),
            'stone_count': stone_count,
            'best_move': best_move_payload,
            'winrate': analysis.get('winrate', 0.5),
            'score_lead': analysis.get('score_lead', 0.0),
            'policy': policy_list,
        }

        return {
            'sym_hash': sym_hash,
            'state_hash': sym_hash,
            'player_to_move': player_to_move,
            'symmetry_index': 0,
            'komi': deep_result['komi'],
            'move_number': deep_result['move_number'],
            'query_id': f"online_{sym_hash}_{int(time.time() * 1000)}",
            'stone_count': stone_count,
            'board_state': board_flat,
            'superko_bans': [0] * (board_state.board_size ** 2),
            'history_context': {
                'board_x_size': board_state.board_size,
                'board_y_size': board_state.board_size,
                'encore_phase': 0,
                'num_turns_this_phase': 0,
                'num_consec_valid_turns': 0,
                'consecutive_ending_passes': 0,
                'assume_multiple_handicap': False,
                'is_past_normal_phase_end': False,
                'is_game_finished': False,
                'presumed_next_move': player_to_move,
            },
            'best_move': best_move_payload,
            'deep_result': deep_result,
            'policy': policy_list,
            'winrate': deep_result['winrate'],
            'score_lead': deep_result['score_lead'],
            'uncertainty_metrics': {
                'combined_score': float(uncertainty),
            },
            'children': [],
        }

    def _serialize_board(self, board_state: BoardState) -> Tuple[List[int], Dict[str, int]]:
        """Flatten board state and compute stone counts."""
        board_flat = board_state.board.astype(int).flatten().tolist()
        black = int((board_state.board == 1).sum())
        white = int((board_state.board == -1).sum())
        stone_count = {
            'black': black,
            'white': white,
            'total': black + white,
        }
        return board_flat, stone_count
    
    def parse_katago_analysis(self, analysis_json: str) -> Optional[Dict[str, Any]]:
        """
        Parse KataGo analysis JSON output.
        
        Returns:
            Dict with 'moveInfos', 'rootInfo', etc. or None on error
        """
        try:
            # KataGo returns multiple JSON objects (one per analysis interval)
            # We want the last one (final analysis)
            lines = analysis_json.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'moveInfos' in data:
                            return data
                    except json.JSONDecodeError:
                        continue
            return None
        except Exception as e:
            logger.error(f"Error parsing KataGo analysis: {e}")
            return None
    
    def extract_policy_from_analysis(self, analysis: Dict[str, Any]) -> np.ndarray:
        """Extract policy distribution from KataGo analysis."""
        board_size = self.board_state.board_size
        policy = np.zeros(board_size * board_size, dtype=np.float32)
        
        if 'moveInfos' not in analysis:
            return policy
        
        for move_info in analysis['moveInfos']:
            move = move_info.get('move')
            prior = move_info.get('prior', 0.0)
            
            if move and move.upper() not in ['PASS', 'RESIGN']:
                # Convert move to index
                col_letter = move[0].upper()
                row_str = move[1:]
                
                col = ord(col_letter) - ord('A')
                if col >= 8:  # Skip 'I'
                    col -= 1
                
                row = board_size - int(row_str)
                
                if 0 <= row < board_size and 0 <= col < board_size:
                    idx = row * board_size + col
                    policy[idx] = prior
        
        # Normalize
        total = policy.sum()
        if total > 0:
            policy /= total
        
        return policy
    
    def run_deep_augmented_search(
        self,
        board_state: BoardState,
        uncertainty: float,
        recursion_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Run deep augmented MCTS search on complex position.
        
        Recursively applies deep search to child positions if they are complex.
        ALWAYS checks RAG cache before running expensive deep search.
        """
        if recursion_depth >= self.max_recursion_depth:
            logger.info(f"  {'  ' * recursion_depth}→ Max recursion depth reached")
            return self._run_standard_search(board_state, visits=self.standard_visits)
        
        # FIRST: Check if this position is already in RAG cache
        position_hash = board_state.get_sym_hash()
        cached = self.query_rag_exact(position_hash)
        
        if cached and recursion_depth > 0:  # Use cache for recursive calls
            logger.info(f"  {'  ' * recursion_depth}→ Cache hit for position: {position_hash[:12]}")
            ctx, ctx_score = cached.get_best_context(
                current_phase=int(np.abs(board_state.board).sum()),
                current_policy=None,
                current_komi=self.config['katago'].get('komi', 7.5),
                current_score_lead=0.0,
                visits_normalizer=self.visits_normalizer,
            )
            if ctx and ctx_score >= self.min_rag_relevance:
                self.stats['rag_hits'] += 1
                self.stats['exact_matches'] += 1
                # Return cached analysis as result
                return {
                    'best_move': ctx['move'],
                    'policy': ctx['policy'],
                    'winrate': ctx['winrate'],
                    'score_lead': ctx['score_lead'],
                    'visits': ctx['deep_visits'],
                    'deep_visits': ctx['deep_visits'],
                    'cached': True,
                    'child_positions': [],
                }
            else:
                logger.info(
                    f"  {'  ' * recursion_depth}→ Cache entry score {ctx_score:.3f} < min {self.min_rag_relevance:.3f}, continuing search"
                )
        
        # Cache miss - run actual deep search
        self.stats['deep_searches'] += 1
        if recursion_depth > 0:
            self.stats['recursive_searches'] += 1
        
        logger.info(f"  {'  ' * recursion_depth}→ Deep search (depth={recursion_depth}, unc={uncertainty:.3f})")
        
        # Run deep MCTS with more visits
        result = self._run_standard_search(board_state, visits=self.deep_visits)
        result['deep_visits'] = self.deep_visits
        result['recursion_depth'] = recursion_depth
        
        # Check child positions for complexity
        if 'child_positions' in result and len(result['child_positions']) > 0:
            enhanced_children = []
            result_policy = result.get(
                'policy',
                np.zeros(self.board_state.board_size ** 2, dtype=np.float32),
            )
            for child in result['child_positions'][:5]:  # Top 5 children
                child_uncertainty = child.get('uncertainty', 0.0)
                
                if child_uncertainty > self.deep_search_threshold:
                    child_board = child.get('position')
                    if not isinstance(child_board, BoardState):
                        continue
                    
                    child_hash = child_board.get_sym_hash()
                    
                    # Query RAG for child
                    cached = self.query_rag_exact(child_hash)
                    
                    if cached:
                        logger.info(f"  {'  ' * (recursion_depth+1)}→ Child cache hit: {child_hash[:12]}")
                        child_phase = int(np.abs(child_board.board).sum())
                        child_policy = child.get('policy')
                        if child_policy is None:
                            child_policy = result_policy
                        ctx, ctx_score = cached.get_best_context(
                            child_phase,
                            child_policy,
                            self.config['katago'].get('komi', 7.5),
                            child.get('score_lead', 0.0),
                            self.visits_normalizer,
                        )
                        if ctx and ctx_score >= self.min_rag_relevance:
                            child['cached_analysis'] = ctx
                            self.stats['rag_hits'] += 1
                            self.stats['exact_matches'] += 1
                        else:
                            logger.info(
                                f"  {'  ' * (recursion_depth+1)}→ Child context discarded (score={ctx_score:.3f} < {self.min_rag_relevance:.3f})"
                            )
                    else:
                        # Recursively analyze child
                        logger.info(f"  {'  ' * (recursion_depth+1)}→ Recursive search for child")
                        child_analysis = self.run_deep_augmented_search(
                            child_board,
                            child_uncertainty,
                            recursion_depth + 1
                        )
                        child['deep_analysis'] = child_analysis
                        
                        # Store child's analysis
                        self.store_position(
                            child_hash,
                            child_analysis['best_move'],
                            child_uncertainty,
                            child_analysis,
                            board_state=child_board,
                        )
                
                enhanced_children.append(child)
            
            result['enhanced_children'] = enhanced_children
        
        return result
    
    def _run_standard_search(self, board_state: BoardState, visits: int = 0) -> Dict[str, Any]:
        """
        Run MCTS search using KataGo with REAL neural network analysis.
        
        Args:
            board_state: Current board state
            visits: Number of visits to use (0 = use default)
        
        Returns:
            Dict with analysis results including child positions with REAL NN priors
        """
        if visits == 0:
            visits = self.standard_visits
        
        try:
            # Set KataGo's maxVisits parameter before search
            self.katago.set_max_visits(visits)
            
            # Use kata-genmove_analyze to get move AND real NN analysis in one call
            move, analysis = self.katago.genmove_analyze("b")
            
            if not move:
                logger.error("genmove_analyze failed to return move")
                return self._create_fallback_analysis()
            
            board_size = board_state.board_size
            
            # Extract REAL policy from analysis
            if analysis and 'moveInfos' in analysis:
                policy = self.extract_policy_from_analysis(analysis)
                logger.debug(f"✓ Using REAL policy from KataGo NN (moveInfos: {len(analysis['moveInfos'])})")
            else:
                # Fallback: uniform policy if analysis fails
                logger.warning("⚠ No analysis data, using uniform policy")
                policy = np.ones(board_size ** 2) / (board_size ** 2)
            
            # Extract REAL data from analysis
            winrate = 0.5
            score_lead = 0.0
            
            if analysis and 'rootInfo' in analysis:
                root = analysis['rootInfo']
                winrate = root.get('winrate', 0.5)
                score_lead = root.get('scoreLead', 0.0)
            
            # Create child positions from REAL moveInfos
            children = []
            
            if analysis and 'moveInfos' in analysis:
                # Use REAL candidate moves from KataGo's analysis
                move_infos = analysis['moveInfos'][:5]  # Top 5 moves
                
                for i, move_info in enumerate(move_infos):
                    child_move = move_info.get('move')
                    if not child_move or child_move.upper() in ['PASS', 'RESIGN']:
                        continue
                    
                    try:
                        # Create child board state
                        child_board = board_state.copy()
                        child_board.play_move(child_move, 1)  # Black move
                        
                        # Extract REAL NN data for this child
                        child_prior = move_info.get('prior', 0.0)
                        child_visits = move_info.get('visits', 0)
                        child_winrate = move_info.get('winrate', 0.5)
                        child_score_lead = move_info.get('scoreLead', 0.0)
                        
                        # Calculate uncertainty using REAL policy (we already have it)
                        # Use the current policy + moveInfos for uncertainty
                        child_uncertainty = self.calculate_uncertainty(policy, move_infos)
                        
                        children.append({
                            'position': child_board,
                            'move': child_move,
                            'uncertainty': child_uncertainty,
                            'visits': child_visits,
                            'winrate': child_winrate,
                            'score_lead': child_score_lead,
                            'prior': child_prior,
                        })
                    except Exception as e:
                        logger.debug(f"Error creating child position: {e}")
                        continue
            else:
                # Fallback: if no analysis, create one child with best move
                logger.warning("No moveInfos available, creating single child with best move")
                try:
                    child_board = board_state.copy()
                    child_board.play_move(move, 1)
                    
                    # Use uniform uncertainty as fallback
                    child_uncertainty = 0.5
                    
                    children.append({
                        'position': child_board,
                        'move': move,
                        'uncertainty': child_uncertainty,
                        'visits': visits,
                        'winrate': winrate,
                        'score_lead': score_lead,
                        'prior': 1.0,
                    })
                except Exception as e:
                    logger.debug(f"Error creating fallback child: {e}")
            
            return {
                'best_move': move,
                'policy': policy,
                'winrate': winrate,
                'score_lead': score_lead,
                'visits': visits,
                'deep_visits': 0,
                'child_positions': children,
                'moveInfos': analysis.get('moveInfos', []) if analysis else [],
            }
        
        except Exception as e:
            logger.error(f"Error in standard search: {e}", exc_info=True)
            return self._create_fallback_analysis()
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Create fallback analysis when KataGo fails."""
        return {
            'best_move': 'pass',
            'policy': np.random.dirichlet(np.ones(self.board_state.board_size ** 2)),
            'winrate': 0.5,
            'score_lead': 0.0,
            'visits': 0,
            'deep_visits': 0,
            'child_positions': [],
        }
    
    def store_position(
        self,
        sym_hash: str,
        move: str,
        uncertainty: float,
        analysis: Dict[str, Any],
        board_state: Optional[BoardState] = None,
    ):
        """Store position analysis in RAG database."""
        board_ref = board_state if board_state is not None else self.board_state
        raw_entry = self._build_raw_entry(sym_hash, move, analysis, board_ref, uncertainty)
        self._append_raw_entry_to_file(raw_entry)

        # Get or create position context
        if sym_hash not in self.position_db:
            self.position_db[sym_hash] = PositionContext(sym_hash=sym_hash)
            self.stats['positions_stored'] += 1
            
            # Also add to ANN index for retrieval
            embedding = analysis['policy'][:64] if len(analysis['policy']) >= 64 else np.pad(analysis['policy'], (0, 64 - len(analysis['policy'])))
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
            
            entry = MemoryEntry.create(
                embed=embedding,
                canonical_board=sym_hash,
                best_moves=[{'move': move, 'prob': 1.0}],
                importance=uncertainty,
            )
            self.rag_index.add(entry)
        
        # Add new context
        pos_ctx = self.position_db[sym_hash]
        pos_ctx.add_context(
            move=move,
            uncertainty=uncertainty,
            deep_visits=analysis.get('deep_visits', 0),
            policy=analysis['policy'],
            winrate=analysis.get('winrate', 0.5),
            score_lead=analysis.get('score_lead', 0.0),
            game_phase=self.stones_on_board,
            metadata={
                'recursion_depth': analysis.get('recursion_depth', 0),
                'move_number': len(board_ref.move_history),
                'komi': self.config['katago'].get('komi', 7.5),
                'raw_entry': raw_entry,
            }
        )
        self.stats['contexts_added'] += 1

    def _append_raw_entry_to_file(self, raw_entry: Dict[str, Any]):
        """Append a raw entry to added_entries.json for persistence."""
        if not self.added_entries_file:
            return
        try:
            self.added_entries_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.added_entries_file, 'a') as f:
                json.dump(raw_entry, f)
                f.write('\n')
            logger.info(f"Appended new deep-search entry to {self.added_entries_file}")
        except Exception as exc:
            logger.error(f"Failed to append raw entry to {self.added_entries_file}: {exc}")
    
    def generate_move(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generate move using recursive deep search with RAG.
        
        Algorithm:
        1. Get analysis of current position
        2. Calculate uncertainty
        3. If high uncertainty:
           a. Create sym_hash for position
           b. Query RAG for exact match
           c. If found: Use best context
           d. If not found: Run deep augmented search (recursive)
           e. Store results in RAG
        """
        start_time = time.time()
        
        # Get analysis for current position
        analysis = self._run_standard_search(self.board_state, visits=self.standard_visits)
        
        if not analysis or 'best_move' not in analysis:
            return "pass", {'error': 'Failed to analyze position'}
        
        move = analysis['best_move'].upper()
        policy = analysis.get('policy', np.zeros(self.board_state.board_size ** 2))
        score_lead = analysis.get('score_lead', 0.0)
        current_komi = self.config['katago'].get('komi', 7.5)
        
        # Extract move info for uncertainty calculation
        move_info = []
        if 'raw_analysis' in analysis and 'moveInfos' in analysis['raw_analysis']:
            move_info = analysis['raw_analysis']['moveInfos']
        
        # Calculate uncertainty
        uncertainty = self.calculate_uncertainty(policy, move_info)
        self.stats['moves'] += 1
        self.stats['total_uncertainty'] += uncertainty
        
        # Check if we should do deep search
        rag_queried = False
        rag_hit = False
        exact_match = False
        deep_searched = False
        contexts_count = 0
        metadata = {
            'uncertainty': uncertainty,
            'rag_queried': False,
            'rag_hit': False,
            'exact_match': False,
            'deep_searched': False,
            'contexts_count': 0,
            'time_ms': 0.0,
            'forced_rag': False,
        }
        
        sym_hash = self.board_state.get_sym_hash()

        forced_rag = (
            self.force_rag_sym is not None and
            sym_hash.lower() == self.force_rag_sym
        )

        if forced_rag:
            logger.info(f"  → Forcing RAG retrieval for {sym_hash}")
            cached = self.query_rag_exact(sym_hash)
            if cached:
                describe = (
                    f"Force contexts={len(cached.contexts)} phase={self.stones_on_board} "
                    f"komi={current_komi} score_lead={score_lead:.2f}"
                )
                logger.info(describe)
                phase_norm = self.stones_on_board / 361.0
                policy_norm = np.linalg.norm(policy) + 1e-9
                for idx, entry in enumerate(cached.contexts):
                    ctx_phase_norm = entry['game_phase'] / 361.0
                    phase_cos = 1.0 - abs(ctx_phase_norm - phase_norm)
                    ctx_policy = entry['policy']
                    policy_cos = float(np.dot(ctx_policy, policy) / ((np.linalg.norm(ctx_policy) + 1e-9) * policy_norm))
                    ctx_meta = entry['metadata'] or {}
                    raw_entry = ctx_meta.get('raw_entry') or {}
                    ctx_komi = raw_entry.get('komi', ctx_meta.get('komi', current_komi))
                    visits_score = min(entry['deep_visits'] / max(self.visits_normalizer, 1.0), 1.0)
                    logger.info(
                        "    Context[%d] move=%s phase=%d deep=%d phase_cos=%.3f policy_cos=%.3f visits=%.3f komi_delta=%.2f",
                        idx,
                        entry['move'],
                        entry['game_phase'],
                        entry['deep_visits'],
                        phase_cos,
                        policy_cos,
                        visits_score,
                        ctx_komi - current_komi,
                    )
                forced_ctx, forced_score = cached.get_best_context(
                    self.stones_on_board,
                    policy,
                    current_komi,
                    score_lead,
                    self.visits_normalizer,
                )
                if forced_ctx and forced_score >= self.min_rag_relevance:
                    move = forced_ctx['move']
                    rag_queried = True
                    rag_hit = True
                    exact_match = True
                    contexts_count = len(cached.contexts)
                    metadata.update({
                        'forced_rag': True,
                        'forced_context': forced_ctx['metadata'].get('source'),
                    })
                    logger.info(
                        "  → Forced RAG move %s (deep_visits=%d, score=%.3f)",
                        move,
                        forced_ctx['deep_visits'],
                        forced_score,
                    )
                    # Continue to board update below
                else:
                    logger.warning(
                        "  → Forced retrieval rejected context (score=%.3f, min=%.3f)",
                        forced_score,
                        self.min_rag_relevance,
                    )
            else:
                logger.warning("  → Forced sym_hash %s not found in RAG cache", sym_hash)

        if metadata['forced_rag']:
            pass  # Skip uncertainty/deep search flow
        elif uncertainty > self.deep_search_threshold:
            logger.info(f"  → Complex position detected (unc={uncertainty:.3f})")
            
            # Query RAG for exact match
            self.stats['rag_queries'] += 1
            rag_queried = True
            
            cached = self.query_rag_exact(sym_hash)
            
            if cached:
                self.stats['rag_hits'] += 1
                rag_hit = True
                contexts_count = len(cached.contexts)
                
                # Get best matching context
                best_ctx, best_score = cached.get_best_context(
                    self.stones_on_board,
                    policy,
                    current_komi,
                    score_lead,
                    self.visits_normalizer,
                )
                
                if best_ctx and best_score >= self.min_rag_relevance:
                    self.stats['exact_matches'] += 1
                    exact_match = True
                    logger.info(f"  → Exact match found: {sym_hash[:12]} ({contexts_count} contexts)")
                    logger.info(
                        f"  → Using cached analysis (deep_visits={best_ctx['deep_visits']}, score={best_score:.3f})"
                    )
                    move = best_ctx['move']
                else:
                    logger.info(
                        f"  → Position found but score {best_score:.3f} < min {self.min_rag_relevance:.3f}, running deep search"
                    )
                    deep_searched = True
                    deep_analysis = self.run_deep_augmented_search(self.board_state, uncertainty, 0)
                    move = deep_analysis['best_move']
                    self.store_position(
                        sym_hash,
                        move,
                        uncertainty,
                        deep_analysis,
                        board_state=self.board_state.copy(),
                    )
            else:
                logger.info(f"  → No match found: {sym_hash[:12]}, running deep search")
                deep_searched = True
                # No match - run deep augmented search
                deep_analysis = self.run_deep_augmented_search(self.board_state, uncertainty, 0)
                move = deep_analysis['best_move']
                # Store results
                self.store_position(
                    sym_hash,
                    move,
                    uncertainty,
                    deep_analysis,
                    board_state=self.board_state.copy(),
                )
        
        # Update game state
        self.move_history.append(move)
        self.board_state.play_move(move, 1)  # Black move
        if move not in ["PASS", "RESIGN"]:
            self.stones_on_board += 1
        
        time_ms = (time.time() - start_time) * 1000
        
        metadata.update({
            'rag_queried': rag_queried,
            'rag_hit': rag_hit,
            'exact_match': exact_match,
            'deep_searched': deep_searched,
            'contexts_count': contexts_count,
            'time_ms': time_ms,
        })
        
        return move, metadata
    
    def update_board_with_opponent_move(self, move: str):
        """Update internal board state with opponent's move."""
        self.board_state.play_move(move, -1)  # White move
        if move not in ["PASS", "RESIGN"]:
            self.stones_on_board += 1


def run_match(
    katago_executable: str,
    katago_model: str,
    katago_config: str,
    config_path: str,
    num_games: int = 1,
    max_moves: int = 200,
    force_rag_sym: Optional[str] = None,
):
    """Run DataGo vs KataGo match with recursive deep search."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 70)
    logger.info("DataGo Bot vs KataGo Match (Recursive Deep Search + RAG)")
    logger.info("=" * 70)
    logger.info(f"Games: {num_games}, Max moves per game: {max_moves}")
    logger.info(f"Deep search threshold: {config['rag_query']['uncertainty_threshold'] + 0.05}")
    logger.info(f"Max recursion depth: {config['recursion']['max_recursion_depth']}")
    logger.info(f"Deep MCTS visits: {config['deep_mcts']['max_visits']}")
    logger.info("")
    
    # Start KataGo GTP
    logger.info("Starting KataGo GTP on GPU 7...")
    cmd = [
        katago_executable,
        'gtp',
        '-model', katago_model,
        '-config', katago_config,
    ]
    katago = GTPController(command=cmd)
    
    # Initialize DataGo player
    datago = RecursiveDataGoPlayer(config, katago, force_rag_sym=force_rag_sym)
    
    results = []
    
    for game_num in range(1, num_games + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Game {game_num}/{num_games}")
        logger.info(f"{'=' * 70}\n")
        
        # Reset for new game
        datago.move_history = []
        datago.stones_on_board = 0
        datago.current_recursion_depth = 0
        datago.board_state = BoardState(19)  # Reset board state
        
        try:
            # Setup game
            katago.boardsize(19)
            katago.clear_board()
            katago.komi(7.5)
            
            move_number = 0
            passes = 0
            
            while move_number < max_moves:
                move_number += 1
                
                if move_number % 2 == 1:  # Black (DataGo)
                    logger.info(f"Move {move_number}: DataGo (Black) thinking...")
                    
                    move, metadata = datago.generate_move()
                    
                    # Log with metadata
                    log_msg = f"Move {move_number}: DataGo (Black) plays {move}"
                    log_msg += f" [unc={metadata['uncertainty']:.3f}"
                    if metadata['deep_searched']:
                        log_msg += ", DEEP"
                    if metadata['exact_match']:
                        log_msg += f", CACHED({metadata['contexts_count']})"
                    log_msg += f", {metadata['time_ms']:.0f}ms]"
                    logger.info(log_msg)
                    
                else:  # White (KataGo)
                    logger.info(f"Move {move_number}: KataGo (White) thinking...")
                    move = katago.genmove("W")
                    
                    if not move:
                        logger.error("Failed to get move")
                        break
                    
                    move = move.upper()
                    logger.info(f"Move {move_number}: KataGo (White) plays {move}")
                    
                    # Update DataGo's board state with opponent move
                    datago.update_board_with_opponent_move(move)
                
                # Check game end
                if move == "RESIGN":
                    winner = "KataGo" if move_number % 2 == 1 else "DataGo"
                    logger.info(f"\n{'DataGo' if move_number % 2 == 1 else 'KataGo'} resigned. {winner} wins!")
                    results.append(winner)
                    break
                elif move == "PASS":
                    passes += 1
                    if passes >= 2:
                        logger.info("\nBoth players passed. Game over.")
                        success, score_result = katago.send_command("final_score")
                        if success and score_result:
                            logger.info(f"Final score: {score_result}")
                            if score_result.startswith('B+'):
                                logger.info("DataGo (Black) wins by scoring!")
                                results.append("DataGo")
                            elif score_result.startswith('W+'):
                                logger.info("KataGo (White) wins by scoring!")
                                results.append("KataGo")
                            else:
                                logger.info("Draw by score")
                                results.append("Draw")
                        else:
                            logger.info("Score calculation failed, recording as fail.")
                            results.append("Fail")
                        break
                else:
                    passes = 0
                
                time.sleep(0.05)
            
            if move_number >= max_moves:
                logger.info(f"\nReached max moves ({max_moves}). Scoring game...")
                # Use KataGo to score the final position
                success, score_result = katago.send_command("final_score")
                if success and score_result:
                    logger.info(f"Final score: {score_result}")
                    # Parse score (format: "B+5.5" or "W+2.5")
                    if score_result.startswith('B+'):
                        logger.info("DataGo (Black) wins by scoring!")
                        results.append("DataGo")
                    elif score_result.startswith('W+'):
                        logger.info("KataGo (White) wins by scoring!")
                        results.append("KataGo")
                    else:
                        logger.info(f"Draw by score")
                        results.append("Draw")
                else:
                    logger.info("Could not determine score. Recording as Draw.")
                    results.append("Draw")
            
            # Print game statistics
            logger.info(f"\nGame {game_num} Statistics:")
            logger.info(f"  Moves played: {datago.stats['moves']}")
            logger.info(f"  RAG queries: {datago.stats['rag_queries']}")
            logger.info(f"  RAG hits: {datago.stats['rag_hits']}")
            logger.info(f"  Exact matches: {datago.stats['exact_matches']}")
            logger.info(f"  Deep searches: {datago.stats['deep_searches']}")
            logger.info(f"  Recursive searches: {datago.stats['recursive_searches']}")
            logger.info(f"  Unique positions: {datago.stats['positions_stored']}")
            logger.info(f"  Total contexts: {datago.stats['contexts_added']}")
            if datago.stats['moves'] > 0:
                avg_unc = datago.stats['total_uncertainty'] / datago.stats['moves']
                logger.info(f"  Average uncertainty: {avg_unc:.3f}")
        
        except Exception as e:
            logger.error(f"Error in game {game_num}: {e}", exc_info=True)
            results.append("Error")
    
    # Cleanup
    logger.info("\nCleaning up...")
    katago.quit()
    
    # Overall statistics
    logger.info("\n" + "=" * 70)
    logger.info("Overall Match Statistics")
    logger.info("=" * 70)
    logger.info(f"Total moves: {datago.stats['moves']}")
    logger.info(f"Total RAG queries: {datago.stats['rag_queries']}")
    logger.info(f"Total exact matches: {datago.stats['exact_matches']}")
    logger.info(f"Total deep searches: {datago.stats['deep_searches']}")
    logger.info(f"Total recursive searches: {datago.stats['recursive_searches']}")
    logger.info(f"Unique positions stored: {datago.stats['positions_stored']}")
    logger.info(f"Total contexts: {datago.stats['contexts_added']}")
    logger.info(f"Avg contexts per position: {datago.stats['contexts_added'] / max(1, datago.stats['positions_stored']):.1f}")
    
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
    parser = argparse.ArgumentParser(description="Run DataGo vs KataGo with recursive deep search")
    parser.add_argument('--katago-executable', required=True, help='Path to KataGo binary')
    parser.add_argument('--katago-model', required=True, help='Path to KataGo model')
    parser.add_argument('--katago-config', required=True, help='KataGo config')
    parser.add_argument('--config', required=True, help='DataGo config.yaml')
    parser.add_argument('--games', type=int, default=1, help='Number of games (default: 1)')
    parser.add_argument('--max-moves', type=int, default=200, help='Max moves per game (default: 200)')
    parser.add_argument('--force-rag-sym', help='Force RAG retrieval for this sym_hash (testing)')
    
    args = parser.parse_args()
    
    run_match(
        katago_executable=args.katago_executable,
        katago_model=args.katago_model,
        katago_config=args.katago_config,
        config_path=args.config,
        num_games=args.games,
        max_moves=args.max_moves,
        force_rag_sym=args.force_rag_sym,
    )


if __name__ == '__main__':
    main()
