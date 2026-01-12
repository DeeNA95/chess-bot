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
        # History (96) + Metadata (8) + Attacks (12 - Unused/Zeros) = 116
        self.num_planes = (self.history_len * self.planes_per_step) + 8 + 12
        self.planes_per_step = 12
        self.shape = (self.num_planes, 8, 8)
        self.device = str(device)


    def encode_node(self, node) -> torch.Tensor:
        """
        Efficiently encodes state from MCTS node by traversing parents.
        Avoids board copying/popping by using the tree structure for history.
        """
        state = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        me = node.board.turn

        # 1. History Planes (0-95)
        current = node
        for t in range(self.history_len):
            base_idx = t * 12
            if current:
                self._encode_pieces(current.board, state, base_idx, me)
                current = current.parent
            else:
                break

        # 2. Metadata Planes
        metadata_idx = self.history_len * self.planes_per_step
        self._encode_metadata(node.board, state, metadata_idx, me)

        # 3. Attack/Defense Maps (Disabled for speed, planes kept as zeros)

        return state

    def encode(self, board: chess.Board) -> torch.Tensor:
        state = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        me = board.turn

        # Working copy for history traversal
        # stack=True is needed to pop moves
        path_board = board.copy(stack=True)

        # 1. History Planes (0-95)
        for t in range(self.history_len):
            base_idx = t * 12
            self._encode_pieces(path_board, state, base_idx, me)

            if t < self.history_len - 1:
                if path_board.move_stack:
                    path_board.pop()
                else:
                    break

        # 2. Metadata Planes
        # Start after history planes
        metadata_idx = self.history_len * self.planes_per_step
        self._encode_metadata(board, state, metadata_idx, me)



        return state

    def _encode_pieces(self, board, state, base_idx, perspective):
        """Encodes 12 planes of pieces for a given board state."""
        me = perspective
        enemy = not me

        # Pre-compute rank/file lookups if we were strict, but python-chess is fast enough here
        # or we optimize _map_sq inlined.

        # My Pieces
        for i, piece_type in enumerate(PIECE_TYPES):
            for sq in board.pieces(piece_type, me):
                # Inline mapping
                r, c = chess.square_rank(sq), chess.square_file(sq)
                if perspective == chess.BLACK:
                    r, c = 7 - r, 7 - c
                state[base_idx + i, r, c] = 1.0

        # Enemy Pieces
        for i, piece_type in enumerate(PIECE_TYPES):
            for sq in board.pieces(piece_type, enemy):
                r, c = chess.square_rank(sq), chess.square_file(sq)
                if perspective == chess.BLACK:
                    r, c = 7 - r, 7 - c
                state[base_idx + 6 + i, r, c] = 1.0

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
            r, c = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            if perspective == chess.BLACK:
                r, c = 7 - r, 7 - c
            state[base_idx+4, r, c] = 1.0

        # Halfmove Clock (clock/50) (101)
        state[base_idx+5, :, :] = min(board.halfmove_clock / 50.0, 1.0)

        # Fullmove Number (num/200) (102)
        state[base_idx+6, :, :] = min(board.fullmove_number / 200.0, 1.0)

        # Color (103) - 1.0 if Black
        if perspective == chess.BLACK:
            state[base_idx+7, :, :] = 1.0



    def _fill_plane_from_bitboard(self, state, plane_idx, bitboard, perspective):
        """Helper to fill a tensor plane from an integer bitboard."""
        while bitboard:
            sq = bitboard & -bitboard # LS1B
            r, c = chess.square_rank(sq.bit_length() - 1), chess.square_file(sq.bit_length() - 1)
            if perspective == chess.BLACK:
                r, c = 7 - r, 7 - c
            state[plane_idx, r, c] = 1.0
            bitboard &= bitboard - 1

    def get_action_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Computes the action mask for a given board state.
        Returns a boolean tensor of shape (4096,).
        """
        mask = torch.zeros(4096, dtype=torch.bool, device=self.device)
        valid_indices = [
            move.from_square * 64 + move.to_square
            for move in board.legal_moves
            if not move.promotion or move.promotion == chess.QUEEN
        ]
        if valid_indices:
            mask[valid_indices] = True
        return mask

