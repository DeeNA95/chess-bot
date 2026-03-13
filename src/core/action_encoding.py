"""
AlphaZero-style action encoding for chess.

Encodes chess moves into a flat action space of size 4672 (73 move types × 64 from-squares).

Move types (73 total):
  - 56 queen-type moves: 7 distances × 8 directions (N, NE, E, SE, S, SW, W, NW)
  - 8 knight moves: 8 possible L-shaped jumps
  - 9 underpromotions: 3 piece types (knight, bishop, rook) × 3 directions (left, forward, right)

Queen promotions are represented as queen-type moves (moving forward 1 square in the
appropriate direction). Underpromotions get their own dedicated move types.

All encoding is done from the perspective of the current player (canonical form).
The board state encoding already flips for Black's perspective, so we flip moves similarly.
"""

import chess
import torch
from typing import Optional

# Action space constants
NUM_MOVE_TYPES = 73
NUM_SQUARES = 64
ACTION_SPACE_SIZE = NUM_MOVE_TYPES * NUM_SQUARES  # 4672

# 8 compass directions as (delta_rank, delta_file)
# Order: N, NE, E, SE, S, SW, W, NW
DIRECTIONS = [
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1), # SW
    (0, -1),  # W
    (1, -1),  # NW
]

# 8 knight move offsets as (delta_rank, delta_file)
KNIGHT_MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]

# Underpromotion directions: (delta_file) for the 3 capture/push directions
# Left capture (-1), forward (0), right capture (+1)
UNDERPROMO_DIRECTIONS = [-1, 0, 1]

# Underpromotion piece types (queen is handled as a queen-type move)
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _flip_square(sq: int) -> int:
    """Flip a square for Black's perspective: mirror rank and file."""
    r, f = divmod(sq, 8)
    return (7 - r) * 8 + (7 - f)


def move_to_action(move: chess.Move, perspective: chess.Color) -> int:
    """
    Encode a chess.Move into an AlphaZero action index.

    Args:
        move: The chess move to encode.
        perspective: The color of the player making the move (board.turn).

    Returns:
        An integer action index in [0, 4672).
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Flip squares for Black's perspective (canonical encoding)
    if perspective == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)

    from_rank, from_file = divmod(from_sq, 8)
    to_rank, to_file = divmod(to_sq, 8)

    dr = to_rank - from_rank
    df = to_file - from_file

    # Check for underpromotions (knight, bishop, rook)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Underpromotion: 9 move types (3 pieces × 3 directions)
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        # Direction: -1 (left capture), 0 (forward), 1 (right capture)
        dir_idx = UNDERPROMO_DIRECTIONS.index(df)
        move_type = 64 + piece_idx * 3 + dir_idx  # 64..72
        return move_type * NUM_SQUARES + from_sq

    # Check for knight moves
    if (dr, df) in KNIGHT_MOVES:
        knight_idx = KNIGHT_MOVES.index((dr, df))
        move_type = 56 + knight_idx  # 56..63
        return move_type * NUM_SQUARES + from_sq

    # Queen-type move (includes queen promotions)
    # Find direction and distance
    if dr == 0 and df == 0:
        # This shouldn't happen for a legal move, but handle gracefully
        return 0

    # Normalize direction
    if dr != 0:
        norm_dr = dr // abs(dr)
    else:
        norm_dr = 0
    if df != 0:
        norm_df = df // abs(df)
    else:
        norm_df = 0

    direction = (norm_dr, norm_df)
    distance = max(abs(dr), abs(df))

    dir_idx = DIRECTIONS.index(direction)
    dist_idx = distance - 1  # 0-indexed (distance 1..7 → index 0..6)

    move_type = dir_idx * 7 + dist_idx  # 0..55
    return move_type * NUM_SQUARES + from_sq


def action_to_move(action_idx: int, board: chess.Board) -> chess.Move:
    """
    Decode an AlphaZero action index back into a chess.Move.

    Args:
        action_idx: An integer action index in [0, 4672).
        board: The current board state (needed for determining perspective and promotion).

    Returns:
        A chess.Move object.
    """
    perspective = board.turn
    move_type = action_idx // NUM_SQUARES
    from_sq = action_idx % NUM_SQUARES

    # Un-flip for Black
    if perspective == chess.BLACK:
        from_sq = _flip_square(from_sq)

    from_rank, from_file = divmod(from_sq if perspective == chess.WHITE else _flip_square(from_sq), 8)
    # Actually we need the canonical from_sq for delta computation
    # Let's recalculate properly
    canonical_from = action_idx % NUM_SQUARES
    can_rank, can_file = divmod(canonical_from, 8)

    promotion = None

    if move_type < 56:
        # Queen-type move
        dir_idx = move_type // 7
        dist_idx = move_type % 7
        distance = dist_idx + 1

        dr, df = DIRECTIONS[dir_idx]
        to_rank = can_rank + dr * distance
        to_file = can_file + df * distance
        to_sq_canonical = to_rank * 8 + to_file

        # Check if this is a promotion (pawn reaching rank 7 in canonical space)
        if to_rank == 7:
            # Check if the piece is a pawn
            actual_from = from_sq
            piece = board.piece_at(actual_from)
            if piece and piece.piece_type == chess.PAWN:
                promotion = chess.QUEEN

    elif move_type < 64:
        # Knight move
        knight_idx = move_type - 56
        dr, df = KNIGHT_MOVES[knight_idx]
        to_rank = can_rank + dr
        to_file = can_file + df
        to_sq_canonical = to_rank * 8 + to_file

    else:
        # Underpromotion
        promo_idx = move_type - 64
        piece_idx = promo_idx // 3
        dir_idx = promo_idx % 3

        promotion = UNDERPROMO_PIECES[piece_idx]
        df = UNDERPROMO_DIRECTIONS[dir_idx]
        to_rank = can_rank + 1  # Pawns always move forward 1 rank for promotion
        to_file = can_file + df
        to_sq_canonical = to_rank * 8 + to_file

    # Un-flip the target square for Black
    if perspective == chess.BLACK:
        to_sq = _flip_square(to_sq_canonical)
    else:
        to_sq = to_sq_canonical

    return chess.Move(from_sq, to_sq, promotion=promotion)


def get_action_mask(board: chess.Board, device: str = "cpu") -> torch.Tensor:
    """
    Compute a boolean action mask for all legal moves.

    Args:
        board: The current board state.
        device: Torch device for the tensor.

    Returns:
        A boolean tensor of shape (4672,) where True indicates a legal move.
    """
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
    perspective = board.turn

    valid_indices = [move_to_action(move, perspective) for move in board.legal_moves]

    if valid_indices:
        mask[valid_indices] = True

    return mask
