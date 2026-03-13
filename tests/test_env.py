import unittest
import torch
import chess
from src.core.chess_env import ChessEnv
from src.core.action_encoding import move_to_action, ACTION_SPACE_SIZE

class TestChessEnv(unittest.TestCase):
    def setUp(self):
        self.env = ChessEnv()

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs['observation'].shape, (116, 8, 8))
        self.assertEqual(obs['action_mask'].shape, (ACTION_SPACE_SIZE,))
        self.assertIsInstance(obs['observation'], torch.Tensor)
        self.assertIsInstance(obs['action_mask'], torch.Tensor)
        # Initial position has 20 legal moves
        self.assertEqual(torch.sum(obs['action_mask']).item(), 20)

    def test_step_pawn_push(self):
        self.env.reset()
        # E2 -> E4 (White's perspective)
        move = chess.Move.from_uci("e2e4")
        action = move_to_action(move, chess.WHITE)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check board state
        self.assertEqual(self.env.board.piece_at(chess.E4).piece_type, chess.PAWN)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0) # No capture, intermediate reward 0

    def test_illegal_move(self):
        self.env.reset()
        # E2 -> E5 (Illegal for white pawn)
        move = chess.Move.from_uci("e2e5")
        action = move_to_action(move, chess.WHITE)

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

        # Move Qh4# (Black's perspective)
        move = chess.Move.from_uci("d8h4")
        action = move_to_action(move, chess.BLACK)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, 1.0)
        self.assertTrue(self.env.board.is_checkmate())

if __name__ == '__main__':
    unittest.main()
