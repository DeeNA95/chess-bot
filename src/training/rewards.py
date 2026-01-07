import chess
import chess.engine
import numpy as np
from typing import Optional

class RewardEngine:
    """
    Calculates rewards for the Chess Agent.
    """

    def __init__(self, stockfish_path: Optional[str] = None):
        self.stockfish_path = stockfish_path
        self.engine = None
        if self.stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception as e:
                print(f"Warning: Could not start Stockfish at {stockfish_path}: {e}")

    def get_reward(self, board: chess.Board, move: chess.Move) -> float:
        """
        Returns the reward for the given move on the board.
        Assumes board is AFTER the move has been pushed.
        """

        # 1. Terminal Rewards
        if board.is_checkmate():
            return 1.0
        if board.is_game_over():
            return 0.0

        reward = 0.0

        # 2. Material Heuristic (Heuristic is applied AFTER move in this logic)
        # We check the board's move stack for the last move
        if board.move_stack:
            last_move = board.move_stack[-1]
            # Since board is already pushed, we'd need to undo to use is_capture
            # Or just trust our Stockfish evaluation for captures.
            # However, for simplicity, let's just check if the last move captured anything.
            # In python-chess, we can check board.is_capture(last_move) by looking at the board BEFORE push.
            # But here we are already pushed.
            # Let's use board.peek() if we want to check.
            pass

        # 3. Stockfish Verification (Dense Reward)
        if self.engine:
            # Analyze position from our perspective (the player who just moved)
            # board.turn is now the opponent.
            info = self.engine.analyse(board, chess.engine.Limit(depth=16))

            # info["score"].relative is for board.turn (the opponent!)
            score_obj = info["score"].relative

            if score_obj.is_mate():
                # If mate is positive for opponent, it's negative for us.
                score = -10000 if score_obj.mate() > 0 else 10000
            else:
                score = -score_obj.score()

            # Normalize and add to reward
            # tanh(score/100) maps [-inf, inf] to [-1, 1]
            reward += np.tanh(score / 100.0) * 0.1

        return reward

    def close(self):
        if self.engine:
            self.engine.quit()
