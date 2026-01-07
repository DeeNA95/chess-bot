import chess
import chess.engine
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union

class ChessVerifier:
    """
    Base class for Chess verifiers.
    Inspired by Prime Intellect's verifiers library.
    """
    def __call__(self, board: chess.Board, move: chess.Move, **kwargs) -> float:
        raise NotImplementedError

    def close(self):
        pass

class StockfishVerifier(ChessVerifier):
    def __init__(self, stockfish_path: str, depth: int = 8):
        self.stockfish_path = stockfish_path
        self.depth = depth
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as e:
            print(f"Warning: StockfishVerifier failed to start: {e}")
            self.engine = None

    def __call__(self, board: chess.Board, move: chess.Move, **kwargs) -> float:
        if not self.engine:
            return 0.0

        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score_obj = info.get("score")
        if score_obj is None:
            return 0.0

        relative_score = score_obj.relative

        if relative_score.is_mate():
            mate_moves = relative_score.mate()
            # If mate_moves is None (shouldn't happen if is_mate is true?), default to 0
            if mate_moves is not None and mate_moves > 0:
                 score = 10000
            else:
                 score = -10000
        else:
            val = relative_score.score()
            score = -val if val is not None else 0 #negated because rewards is called after board move so will be at opponent

        return float(np.tanh(score / 100.0))

    def close(self):
        if self.engine:
            self.engine.quit()

class MaterialVerifier(ChessVerifier):
    VALUE_MAP = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    def __call__(self, board: chess.Board, move: chess.Move, **kwargs) -> float:
        captured_piece = kwargs.get('captured_piece')
        if captured_piece:
            return float(self.VALUE_MAP.get(captured_piece, 0) * 0.01 )
        return 0.0

class OutcomeVerifier(ChessVerifier):
    def __call__(self, board: chess.Board, move: chess.Move, **kwargs) -> float:
        if board.is_checkmate():
            return 1.0
        return 0.0

class ChessRubric:
    def __init__(self):
        self.verifiers: List[tuple[Union[ChessVerifier, Any], float]] = []

    def add_verifier(self, verifier: Union[ChessVerifier, Any], weight: float = 1.0):
        self.verifiers.append((verifier, weight))

    def calculate_reward(self, board: chess.Board, move: chess.Move, **kwargs) -> float:
        total_reward = 0.0
        for verifier, weight in self.verifiers:
            # Check if callable for safe sequential execution
            if callable(verifier):
                reward = verifier(board, move, **kwargs)
                total_reward += reward * weight
        return total_reward

    def calculate_reward_batch(self, boards: List[chess.Board], moves: List[chess.Move], infos: List[Dict]) -> List[float]:
        total_rewards = [0.0] * len(boards)

        # Optimize: get FENs once if needed?
        fens = None

        for verifier, weight in self.verifiers:
            if hasattr(verifier, 'verify_batch'):
                if fens is None:
                    fens = [b.fen() for b in boards]

                rewards = verifier.verify_batch(fens)
                for i, r in enumerate(rewards):
                    total_rewards[i] += r * weight
            elif callable(verifier):
                # Sequential fallback
                for i, (b, m, info) in enumerate(zip(boards, moves, infos)):
                    total_rewards[i] += verifier(b, m, **info) * weight

        return total_rewards

    def close(self):
        for verifier, weight in self.verifiers:
             if hasattr(verifier, 'close'):
                 verifier.close()
