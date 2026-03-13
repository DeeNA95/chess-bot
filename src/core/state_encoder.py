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
        # History (96) + Metadata (8) + Attack/Defense (12) = 116
        self.num_planes = (self.history_len * self.planes_per_step) + 8 + 12
        self.shape = (self.num_planes, 8, 8)
        self.device = str(device)


    def encode_node(self, node) -> torch.Tensor:
        """
        Efficiently encodes state from MCTS node by traversing parents.
        Avoids board copying/popping by using the tree structure for history.
        When history runs out, duplicates the earliest available position.
        """
        state = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        me = node.board.turn

        # 1. History Planes (0-95)
        current = node
        for t in range(self.history_len):
            base_idx = t * 12
            self._encode_pieces(current.board, state, base_idx, me)
            if current.parent is not None:
                current = current.parent
            # else: keep encoding same board for remaining slots

        # 2. Metadata Planes
        metadata_idx = self.history_len * self.planes_per_step
        self._encode_metadata(node.board, state, metadata_idx, me)

        # 3. Attack/Defense Maps
        attack_idx = metadata_idx + 8
        self._encode_attack_defense(node.board, state, attack_idx, me)

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

            if t < self.history_len - 1 and path_board.move_stack:
                path_board.pop()
            # else: keep encoding same board for remaining slots

        # 2. Metadata Planes
        metadata_idx = self.history_len * self.planes_per_step
        self._encode_metadata(board, state, metadata_idx, me)

        # 3. Attack/Defense Maps
        attack_idx = metadata_idx + 8
        self._encode_attack_defense(board, state, attack_idx, me)

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

    def _bitboard_to_plane(self, bb: int, perspective) -> np.ndarray:
        """Convert an integer bitboard to an 8x8 numpy array with perspective flip."""
        # Convert 64-bit integer to 8 bytes, big-endian for MSB-first unpack
        # chess bitboards: bit 0 = a1, bit 63 = h8
        # np.unpackbits with big-endian byte order gives us MSB first,
        # so we reverse the byte order to get LSB-first (a1 first)
        bb_bytes = bb.to_bytes(8, byteorder='little')
        bits = np.unpackbits(np.frombuffer(bb_bytes, dtype=np.uint8))
        # Now bits[0] = bit 0 of first byte = a1, bits[7] = h1, etc.
        plane = bits.reshape(8, 8).astype(np.float32)
        if perspective == chess.BLACK:
            plane = plane[::-1, ::-1].copy()
        return plane

    def _encode_attack_defense(self, board, state, base_idx, perspective):
        """Encode 12 attack/defense planes using bitboard OR + numpy conversion.

        Planes base_idx+0..5:  Squares attacked by my [P,N,B,R,Q,K]
        Planes base_idx+6..11: Squares attacked by enemy [P,N,B,R,Q,K]
        """
        me = perspective
        enemy = not me

        # My attacks (6 planes)
        for i, piece_type in enumerate(PIECE_TYPES):
            combined_attacks = 0
            for sq in board.pieces(piece_type, me):
                combined_attacks |= board.attacks_mask(sq)
            if combined_attacks:
                plane = self._bitboard_to_plane(combined_attacks, perspective)
                state[base_idx + i] = torch.from_numpy(plane)

        # Enemy attacks (6 planes)
        for i, piece_type in enumerate(PIECE_TYPES):
            combined_attacks = 0
            for sq in board.pieces(piece_type, enemy):
                combined_attacks |= board.attacks_mask(sq)
            if combined_attacks:
                plane = self._bitboard_to_plane(combined_attacks, perspective)
                state[base_idx + 6 + i] = torch.from_numpy(plane)

    def get_action_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Computes the action mask for a given board state.
        Returns a boolean tensor of shape (4672,) using AlphaZero action encoding.
        """
        from src.core.action_encoding import get_action_mask
        return get_action_mask(board, device=self.device)


