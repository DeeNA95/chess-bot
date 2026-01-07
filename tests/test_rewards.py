import unittest
import chess
from src.training.rewards import RewardEngine
import shutil

class TestRewardEngine(unittest.TestCase):
    def setUp(self):
        # Find Stockfish path
        self.stockfish_path = shutil.which("stockfish")
        self.reward_engine = RewardEngine(stockfish_path=self.stockfish_path)

    def tearDown(self):
        self.reward_engine.close()

    def test_checkmate_reward(self):
        # Fool's Mate setup: 1. f3 e5 2. g4 Qh4#
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        move = chess.Move.from_uci("d8h4")
        board.push(move)

        # Now the board is in checkmate
        reward = self.reward_engine.get_reward(board, move)
        self.assertEqual(reward, 1.0)

    def test_capture_reward(self):
        if not self.stockfish_path:
            self.skipTest("Stockfish not found")

        # 1. e4 d5 2. exd5
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        move = chess.Move.from_uci("e4d5")
        board.push(move)

        reward = self.reward_engine.get_reward(board, move)
        # Winning a pawn should be positive
        self.assertGreater(reward, 0.0)

    def test_stockfish_comparison(self):
        if not self.stockfish_path:
            self.skipTest("Stockfish not found")

        # Obviously good vs obviously bad move
        # Position: 1. e4. Good: e5. Bad: f6
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

        good_move = chess.Move.from_uci("e7e5")
        bad_move = chess.Move.from_uci("f7f6")

        b1 = board.copy()
        b1.push(good_move)
        good_reward = self.reward_engine.get_reward(b1, good_move)

        b2 = board.copy()
        b2.push(bad_move)
        bad_reward = self.reward_engine.get_reward(b2, bad_move)

        print(f"Good move (e5) reward: {good_reward}")
        print(f"Bad move (f6) reward: {bad_reward}")

        self.assertGreater(good_reward, bad_reward)

if __name__ == '__main__':
    unittest.main()
