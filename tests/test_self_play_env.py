import unittest
import torch
import chess
from src.core.self_play_env import SelfPlayChessEnv
from src.core.action_encoding import move_to_action, ACTION_SPACE_SIZE


class TestSelfPlayChessEnv(unittest.TestCase):
    def setUp(self):
        self.env = SelfPlayChessEnv()

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs['observation'].shape, (116, 8, 8))
        self.assertEqual(obs['action_mask'].shape, (ACTION_SPACE_SIZE,))
        self.assertEqual(info['turn'], chess.WHITE)
        # Initial position has 20 legal moves
        self.assertEqual(torch.sum(obs['action_mask']).item(), 20)

    def test_perspective_alternates(self):
        """Verify that turn alternates after each move."""
        self.env.reset()

        # White's turn
        self.assertEqual(self.env.current_player, chess.WHITE)

        # E2-E4 for White
        move = chess.Move.from_uci("e2e4")
        action = move_to_action(move, chess.WHITE)
        obs, reward, done, _, info = self.env.step(action)

        # Now Black's turn
        self.assertEqual(self.env.current_player, chess.BLACK)
        self.assertEqual(info['mover_color'], chess.WHITE)

        # E7-E5 for Black
        move = chess.Move.from_uci("e7e5")
        action = move_to_action(move, chess.BLACK)
        obs, reward, done, _, info = self.env.step(action)

        # Back to White
        self.assertEqual(self.env.current_player, chess.WHITE)
        self.assertEqual(info['mover_color'], chess.BLACK)

    def test_checkmate_rewards(self):
        """Test that checkmate gives +1 to winner."""
        self.env.reset()
        # Set up Fool's Mate position
        self.env.board = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2')

        # Black delivers Qh4#
        move = chess.Move.from_uci("d8h4")
        action = move_to_action(move, chess.BLACK)
        obs, reward, done, _, info = self.env.step(action)

        self.assertTrue(done)
        self.assertEqual(reward, 1.0)  # Winner (Black) gets +1
        self.assertEqual(info['mover_color'], chess.BLACK)
        self.assertIn('loser_reward', info)
        self.assertEqual(info['loser_reward'], -1.0)

    def test_draw_rewards(self):
        """Test that draw gives 0 to both sides."""
        self.env.reset()
        # Set up a position where White can deliver stalemate
        # White King on a6, Black King on a8, White Queen on h1
        self.env.board = chess.Board('k7/8/1K6/8/8/8/8/7Q w - - 0 1')
        # Move Qa1 which is a legal quiet move (not stalemate)
        # Actually let's find a real stalemate delivery:
        # Position: Ka6, Kb8 forced, Qb6 stalemate
        self.env.board = chess.Board('k7/8/K7/8/8/8/8/1Q6 w - - 0 1')
        # Qb6 creates stalemate for Black king on a8
        move = chess.Move.from_uci("b1b6")
        action = move_to_action(move, chess.WHITE)
        obs, reward, done, _, info = self.env.step(action)

        # This should be stalemate = draw
        self.assertTrue(done)
        self.assertEqual(reward, 0.0)

    def test_illegal_move_penalty(self):
        """Test that illegal moves get severe penalty."""
        self.env.reset()
        # Try illegal move: e2 -> e5 (pawn can't jump 3 squares)
        move = chess.Move.from_uci("e2e5")
        action = move_to_action(move, chess.WHITE)
        obs, reward, done, _, info = self.env.step(action)

        self.assertTrue(done)
        self.assertEqual(reward, -10.0)
        self.assertEqual(info['error'], 'illegal_move')

    def test_captured_piece_in_info(self):
        """Test that captured piece is recorded in info."""
        self.env.reset()
        # Set up a position with capture available
        self.env.board = chess.Board('rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')

        # exd5 (White captures)
        move = chess.Move.from_uci("e4d5")
        action = move_to_action(move, chess.WHITE)
        obs, reward, done, _, info = self.env.step(action)

        self.assertEqual(info['captured_piece'], chess.PAWN)


if __name__ == '__main__':
    unittest.main()
