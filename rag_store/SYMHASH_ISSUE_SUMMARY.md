# Sym Hash Mismatch - Root Cause Analysis

## The Problem
Sym_hashes from original RAG data don't match reconstructed data (0.46% match rate).

## Key Finding
**moves_history is incomplete** - it contains fewer moves than move_number indicates.

## Evidence

### Pattern Observed
```
Game 1: move 3  has 1 move   (missing moves 1-2)
Game 1: move 8  has 6 moves  (missing moves 1-2)  
Game 1: move 13 has 11 moves (missing moves 1-2)

Game 2: move 13 has 0 moves  (missing moves 1-13!)
Game 2: move 18 has 5 moves  (missing moves 1-13)
Game 2: move 23 has 10 moves (missing moves 1-13)
```

**The number of missing moves varies by game** but is **constant within each game**.

## Code Flow

1. **Every move** in self-play:
   - Bot performs MCTS search
   - During search: `datago_collect_search_states()` called (search.cpp:626)
   - After search: Move is made
   - After move: `datago_record_move()` called (play.cpp:1670)

2. **Storage decision** (stats_search.cpp:143):
   ```cpp
   bool if_uncertain(double combined) {
       static thread_local int counter = 0;
       counter++;
       return (counter % 5) == 0;  // Store every 5th search
   }
   ```

3. **When position is flagged**:
   ```cpp
   moveData.move_number = moveNumber;  // From rootHistory.moveHistory.size()
   moveData.moves_history = currentGameRAGData.moves_history;  // Copy
   ```

## The Mystery

**Why are the first N moves missing from moves_history?**

Options:
1. `datago_record_move()` not called for first N moves?
2. `currentGameRAGData.moves_history` cleared/reset partway through game?
3. Some initialization delay?
4. Thread-local state from previous game?

## Impact

When `game_analyzer.py` reconstructs positions:
- It replays `moves_history` 
- This creates a board position that is N moves EARLIER than the original
- KataGo computes sym_hash for this DIFFERENT position
- Result: Hashes don't match

## Next Steps

Need to investigate why moves aren't being recorded from the start of each game.
