import unittest
import torch
import chess
from src.core.chess_env import ChessEnv

class TestChessEnv(unittest.TestCase):
    def setUp(self):
        self.env = ChessEnv()

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs['observation'].shape, (20, 8, 8))
        self.assertEqual(obs['action_mask'].shape, (4096,))
        self.assertIsInstance(obs['observation'], torch.Tensor)
        self.assertIsInstance(obs['action_mask'], torch.Tensor)
        # Initial position has 20 legal moves
        self.assertEqual(torch.sum(obs['action_mask']).item(), 20)

    def test_step_pawn_push(self):
        self.env.reset()
        # E2 (12) -> E4 (28)
        # Action = 12 * 64 + 28 = 796
        action = 12 * 64 + 28

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check board state
        self.assertEqual(self.env.board.piece_at(chess.E4).piece_type, chess.PAWN)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0) # No capture, intermediate reward 0

    def test_illegal_move(self):
        self.env.reset()
        # E2 -> E5 (Illegal for white pawn)
        # Action = 12 * 64 + 36
        action = 12 * 64 + 36

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Expect huge penalty
        self.assertEqual(reward, -10.0)
        self.assertEqual(info['error'], 'illegal_move')
        # Board should not have changed (roughly, or at least e5 is empty)
        self.assertIsNone(self.env.board.piece_at(chess.E5))

    def test_checkmate_reward(self):
        # Fools mate or similar
        self.env.reset()
        # Position where Qh4# is mate
        # 1. f3 e5 2. g4 ??
        # Black (to move) has Queen on d8.
        self.env.board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")

        # Move Qh4#
        # From d8 (59) to h4 (31)
        action = 59 * 64 + 31

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, 1.0)
        self.assertTrue(self.env.board.is_checkmate())

if __name__ == '__main__':
    unittest.main()
