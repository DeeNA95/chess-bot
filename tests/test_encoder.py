import unittest
import chess
import torch
import numpy as np
from src.core.state_encoder import StateEncoder

class TestStateEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = StateEncoder()

    def test_starting_board_shape(self):
        board = chess.Board()
        encoded = self.encoder.encode(board)
        self.assertEqual(encoded.shape, (20, 8, 8))
        self.assertIsInstance(encoded, torch.Tensor)

    def test_white_pieces(self):
        # White Pawn on E2 (Rank 1, File 4)
        # Should be in plane 0
        board = chess.Board()
        encoded = self.encoder.encode(board)
        # Plane 0 = My Pawns. White plays. My Pawns are on Rank 1.
        self.assertEqual(encoded[0, 1, 4], 1.0)

    def test_canonical_perspective_black(self):
        # Black to move
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        encoded = self.encoder.encode(board)

        # Black is playing. Canonical view means Black's pieces should be at "bottom" (Rank 0/1).
        # In FEN, Black pawns are at Rank 6 (index 6).
        # After flipping, they should be at Rank 1.

        # Plane 0 = My Pawns (Black's Pawns)
        # Check A7 (Rank 6, File 0) -> Flipped -> Rank 1, File 7 (if rotated 180) or File 0 (if mirrored)?
        # Implementation said: r = 7 - r, c= 7 - c.
        # So Rank 6 -> Rank 1.
        # File 0 -> File 7.

        # Original Black Pawn at A7 (6, 0).
        # Encoded at (1, 7).
        self.assertEqual(encoded[0, 1, 7], 1.0)

if __name__ == '__main__':
    unittest.main()
