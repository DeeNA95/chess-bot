import torch
import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

class StateEncoder:
    """
    Encodes a chess.Board into a canonical 8x8xN torch.Tensor.

    New Representation (116 Planes):
    0-95:    History (Last 8 positions), 12 planes each [P,N,B,R,Q,K] x [Me, Enemy]
    96-99:   Castling Rights (My K, My Q, Enemy K, Enemy Q)
    100:     En Passant Target
    101:     Half-move clock
    102:     Full-move number
    103:     Color (0=White, 1=Black)
    104-109: Attack Maps (Me: P,N,B,R,Q,K)
    110-115: Defense Maps (Enemy: P,N,B,R,Q,K)
    """

    def __init__(self, device="cpu"):
        self.history_len = 8
        self.planes_per_step = 12
        self.num_planes = (self.history_len * self.planes_per_step) + 8 + 12 # 96 + 8 + 12 = 116
        self.shape = (self.num_planes, 8, 8)
        self.device = device

    def encode(self, board: chess.Board) -> torch.Tensor:
        state = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        me = board.turn

        # 1. History Planes (0-95)
        # We need to traverse back up to history_len - 1 times.
        # We assume the board object has the move stack.
        # To strictly preserve the board state without copying, we will pop and then push back.

        moves_popped = []

        # We need to capture 8 states: T, T-1, ..., T-7
        current_idx = 0

        for t in range(self.history_len):
            base_idx = t * 12

            # Encode pieces for current state
            self._encode_pieces(board, state, base_idx, me)

            # Prepare for next iteration (previous state)
            if t < self.history_len - 1: # Don't pop on the last step if not needed
                if board.move_stack:
                    move = board.pop()
                    moves_popped.append(move)
                else:
                    # No more history available
                    break

        # Restore board state
        for move in reversed(moves_popped):
            board.push(move)

        # 2. Metadata Planes (96-103)
        # Uses the CURRENT board state (original board, now restored)
        self._encode_metadata(board, state, 96, me)

        # 3. Attack/Defense Maps (104-115)
        self._encode_attacks(board, state, 104, me)

        return state

    def _encode_pieces(self, board, state, base_idx, perspective):
        """Encodes 12 planes of pieces for a given board state."""
        me = perspective
        enemy = not me

        # My Pieces
        for i, piece_type in enumerate(PIECE_TYPES):
            # 0-5
            bb = board.pieces_mask(piece_type, me)
            self._fill_plane_from_bitboard(state, base_idx + i, bb, perspective)

        # Enemy Pieces
        for i, piece_type in enumerate(PIECE_TYPES):
            # 6-11
            bb = board.pieces_mask(piece_type, enemy)
            self._fill_plane_from_bitboard(state, base_idx + 6 + i, bb, perspective)

    def _encode_metadata(self, board, state, base_idx, perspective):
        me = perspective
        enemy = not me

        # Castling (96-99)
        if board.has_kingside_castling_rights(me):
            state[base_idx, :, :] = 1.0
        if board.has_queenside_castling_rights(me):
            state[base_idx+1, :, :] = 1.0
        if board.has_kingside_castling_rights(enemy):
            state[base_idx+2, :, :] = 1.0
        if board.has_queenside_castling_rights(enemy):
            state[base_idx+3, :, :] = 1.0

        # En Passant (100)
        if board.ep_square is not None:
            r, c = self._map_sq(board.ep_square, perspective)
            state[base_idx+4, r, c] = 1.0

        # Halfmove Clock (clock/50) (101)
        state[base_idx+5, :, :] = min(board.halfmove_clock / 50.0, 1.0)

        # Fullmove Number (num/200) (102)
        state[base_idx+6, :, :] = min(board.fullmove_number / 200.0, 1.0)

        # Color (103) - 1.0 if Black
        if perspective == chess.BLACK:
            state[base_idx+7, :, :] = 1.0

    def _encode_attacks(self, board, state, base_idx, perspective):
        """
        Computes attack maps - optimized with reduce for bitwise OR.
        """
        from functools import reduce
        from operator import or_

        me = perspective
        enemy = not me

        # 104-109: My Attacks (P,N,B,R,Q,K)
        for i, pt in enumerate(PIECE_TYPES):
            attacks = [int(board.attacks(sq)) for sq in board.pieces(pt, me)]
            attacks_bb = reduce(or_, attacks, 0) if attacks else 0
            if attacks_bb:
                self._fill_plane_from_bitboard(state, base_idx + i, attacks_bb, perspective)

        # 110-115: Enemy Attacks
        for i, pt in enumerate(PIECE_TYPES):
            attacks = [int(board.attacks(sq)) for sq in board.pieces(pt, enemy)]
            attacks_bb = reduce(or_, attacks, 0) if attacks else 0
            if attacks_bb:
                self._fill_plane_from_bitboard(state, base_idx + 6 + i, attacks_bb, perspective)

    def _fill_plane_from_bitboard(self, state, plane_idx, bitboard, perspective):
        """Helper to fill a tensor plane from an integer bitboard."""
        while bitboard:
            sq = bitboard & -bitboard # LS1B
            r, c = self._map_sq(sq.bit_length() - 1, perspective)
            state[plane_idx, r, c] = 1.0
            bitboard &= bitboard - 1

    def _map_sq(self, sq: int, perspective: chess.Color):
        """
        Maps a square integer (0-63) to (row, col) coordinates.
        If perspective is BLACK, flips the board (row = 7 - row).
        """
        r = chess.square_rank(sq)
        c = chess.square_file(sq)

        if perspective == chess.BLACK:
            r = 7 - r
            c = 7 - c

        return r, c
