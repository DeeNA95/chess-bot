"""
Monte Carlo Tree Search (MCTS) guided by neural network.
Implements AlphaZero-style search for improved move selection.
"""

import math
import chess
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    board: chess.Board
    parent: Optional['MCTSNode'] = None
    move: Optional[chess.Move] = None  # Move that led to this node
    prior: float = 0.0  # Prior probability from policy network

    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)  # action_idx -> child
    is_expanded: bool = False

    @property
    def value(self) -> float:
        """Average value across visits."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes

        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploitation + exploration


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses policy network for move priors and value network
    for position evaluation.
    """

    def __init__(
        self,
        model,
        encoder,
        device: str = 'cpu',
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    def search(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """
        Run MCTS from the given position.

        Returns:
            policy: Improved policy (visit counts normalized) as 4096-dim tensor
            value: Root value estimate
        """
        root = MCTSNode(board=board.copy())
        self._expand_node(root)

        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # Selection: traverse to leaf using UCB
            while node.is_expanded and node.children:
                node = self._select_child(node)
                path.append(node)

            # Check terminal
            if node.board.is_game_over():
                value = self._terminal_value(node.board)
            else:
                # Expansion & Evaluation
                if not node.is_expanded:
                    value = self._expand_node(node)
                else:
                    value = 0.0

            # Backpropagation
            self._backpropagate(path, value)

        # Extract policy from visit counts
        policy = self._get_policy(root)
        root_value = root.value

        return policy, root_value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(self.c_puct, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand_node(self, node: MCTSNode) -> float:
        """Expand node using neural network. Returns value estimate."""
        board = node.board

        # Get network predictions
        obs = self.encoder.encode(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs)

        # Get valid moves and their priors
        policy_probs = torch.softmax(logits[0], dim=0).cpu().numpy()

        for move in board.legal_moves:
            # Skip non-Queen promotions
            if move.promotion and move.promotion != chess.QUEEN:
                continue

            action_idx = move.from_square * 64 + move.to_square
            prior = policy_probs[action_idx]

            # Create child node
            child_board = board.copy()
            child_board.push(move)

            child = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior=prior,
            )
            node.children[action_idx] = child

        node.is_expanded = True
        return value.item()

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Backpropagate value up the path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective

    def _terminal_value(self, board: chess.Board) -> float:
        """Get value for terminal position."""
        if board.is_checkmate():
            # Current player is checkmated = loss = -1
            return -1.0
        # Draw
        return 0.0

    def _get_policy(self, root: MCTSNode) -> torch.Tensor:
        """Convert visit counts to policy distribution."""
        policy = torch.zeros(4096, dtype=torch.float32, device=self.device)

        if not root.children:
            return policy

        # Collect visit counts
        total_visits = 0
        for action_idx, child in root.children.items():
            policy[action_idx] = child.visit_count
            total_visits += child.visit_count

        if total_visits > 0:
            if self.temperature == 0:
                # Deterministic: pick best
                best_action = policy.argmax()
                policy = torch.zeros(4096, dtype=torch.float32, device=self.device)
                policy[best_action] = 1.0
            else:
                # Apply temperature
                policy = policy ** (1.0 / self.temperature)
                policy = policy / policy.sum()

        return policy

    def select_move(self, board: chess.Board) -> Tuple[chess.Move, torch.Tensor, float]:
        """
        Run MCTS and select a move.

        Returns:
            move: Selected move
            policy: MCTS policy tensor (for training)
            value: Position value estimate
        """
        policy, value = self.search(board)

        # Sample from policy (during training) or pick best (during play)
        if self.temperature > 0:
            action_idx = torch.multinomial(policy, 1).item()
        else:
            action_idx = policy.argmax().item()

        from_sq = action_idx // 64
        to_sq = action_idx % 64
        move = chess.Move(from_sq, to_sq)

        # Handle promotion
        if chess.square_rank(to_sq) in [0, 7]:
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                move.promotion = chess.QUEEN

        return move, policy, value
