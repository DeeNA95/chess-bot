import torch
import chess

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

class StateEncoder:
    """
    Encodes a chess.Board into a canonical 8x8xN torch.Tensor.

    Canonical Perspective:
    - The 'Current Player' is always viewed as playing 'Up' (White's perspective).
    - If it is Black's turn, we flip the board vertically before encoding.
    - This allows the network to always play "Forward".

    Representation (20 Planes):
    0-5:   My Pieces [P, N, B, R, Q, K]
    6-11:  Enemy Pieces [P, N, B, R, Q, K]
    12-15: Castling Rights (My K, My Q, Enemy K, Enemy Q)
    16:    En Passant Target (One-hot)
    17:    Half-move clock (normalized by 50)
    18:    Full-move number (normalized by 200, clipped)
    19:    Color (0 for White, 1 for Black) - Optional hint
    """

    def __init__(self, device="cpu"):
        self.shape = (20, 8, 8)
        self.device = device

    def encode(self, board: chess.Board) -> torch.Tensor:
        state = torch.zeros(self.shape, dtype=torch.float32, device=self.device)

        # used to determine my pieces
        me = board.turn
        enemy = not me

        # 0-5: My Pieces
        for i, piece_type in enumerate(PIECE_TYPES):
            squares = board.pieces(piece_type, me)
            for sq in squares:
                r, c = self._map_sq(sq, me)
                state[i, r, c] = 1.0

        # 6-11: Enemy Pieces
        for i, piece_type in enumerate(PIECE_TYPES):
            squares = board.pieces(piece_type, enemy)
            for sq in squares:
                r, c = self._map_sq(sq, me)
                state[6 + i, r, c] = 1.0

        # 12-15: Castling Rights
        # My Kingside
        if board.has_kingside_castling_rights(me):
            state[12, :, :] = 1.0
        # My Queenside
        if board.has_queenside_castling_rights(me):
            state[13, :, :] = 1.0
        # Enemy Kingside
        if board.has_kingside_castling_rights(enemy):
            state[14, :, :] = 1.0
        # Enemy Queenside
        if board.has_queenside_castling_rights(enemy):
            state[15, :, :] = 1.0

        # 16: En Passant
        if board.ep_square is not None:
            r, c = self._map_sq(board.ep_square, me)
            state[16, r, c] = 1.0

        # 17: Half move clock (50 move rule)
        state[17, :, :] = min(board.halfmove_clock / 50.0, 1.0)

        # 18: Full move number (game length)
        state[18, :, :] = min(board.fullmove_number / 200.0, 1.0)

        # 19: Color (0 for White, 1 for Black) - useful if tactics differ slightly
        if me == chess.BLACK:
            state[19, :, :] = 1.0

        return state

    def _map_sq(self, sq: int, perspective: chess.Color):
        """
        Maps a square integer (0-63) to (row, col) coordinates.
        If perspective is BLACK, flips the board (row = 7 - row).
        """
        r = chess.square_rank(sq)
        c = chess.square_file(sq)

        if perspective == chess.BLACK:
            # flip ranks and files if black ie 0,0 become 7,7
            # in black bottom left is h8 while a1 for ehite
            r = 7 - r
            c = 7 - c

        return r, c
