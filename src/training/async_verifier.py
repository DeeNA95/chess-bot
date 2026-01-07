import concurrent.futures
import chess
import chess.engine
import numpy as np
from typing import List, Optional

# Global worker state
_worker_engine = None
_worker_depth = 5

def _init_worker(stockfish_path: str, depth: int):
    global _worker_engine, _worker_depth
    _worker_depth = depth
    try:
        _worker_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _worker_engine.configure({"Threads": 1, "Hash": 16})
    except Exception as e:
        print(f"Worker failed to start Stockfish: {e}")
        _worker_engine = None

def _analyze_fen(fen: str) -> float:
    """
    Analyzes FEN and returns normalized reward.
    Executed in worker process.
    """
    if _worker_engine is None:
        return 0.0

    try:
        board = chess.Board(fen)

        info = _worker_engine.analyse(board, chess.engine.Limit(depth=_worker_depth))
        score_obj = info.get("score")
        if score_obj is None:
            return 0.0

        relative_score = score_obj.relative
        if relative_score.is_mate():
            mate_moves = relative_score.mate()
            # Mate in X. Positive is good for side to move.
            if mate_moves is not None and mate_moves > 0:
                 score = 10000
            else:
                 score = -10000
        else:
            # Score can be None if is_mate is True, but we checked is_mate.
            # safe fallback:
            val = relative_score.score()
            score = val if val is not None else 0

        # Normalize: tanh(score/100)
        # Note: relative_score is from the perspective of the side to move (board.turn).
        # We want to return reward for the PREVIOUS move (the agent).
        # But 'fen' represents the state AFTER the agent moved.
        # So "side to move" is the OPPONENT.
        # If score is good for opponent, it's bad for agent.
        # So we should negate the score.

        return float(np.tanh(-score / 100.0))

    except Exception as e:
        return 0.0

class AsyncStockfishVerifier:
    """
    Manages a pool of Stockfish processes for parallel/batch verification.
    """
    def __init__(self, stockfish_path: str, num_workers: int = 8, depth: int = 5):
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(stockfish_path, depth)
        )
        self.num_workers = num_workers

    def verify_batch(self, fens: List[str]) -> List[float]:
        """
        Submits a batch of FENs for analysis.
        Returns a list of rewards (aligned with input).
        """
        results = list(self.executor.map(_analyze_fen, fens))
        return results

    def close(self):
        self.executor.shutdown(wait=False)
