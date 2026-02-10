"""
Monte Carlo Tree Search (MCTS) guided by neural network.
Implements Batched AlphaZero-style search for high-throughput self-play.
"""

import math
import chess
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    board: Optional[chess.Board] = None  # Lazy loaded
    parent: Optional['MCTSNode'] = None
    move: Optional[chess.Move] = None    # Move that led to this node
    prior: float = 0.0                   # Prior probability from policy network

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
    Batched Monte Carlo Tree Search.

    Runs MCTS for multiple games in parallel to maximize GPU utilization
    during neural network inference.
    """

    def __init__(
        self,
        model,
        encoder,
        device: str = 'cpu',
        num_simulations: int = 50,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.0,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search_batch(self, boards: List[chess.Board]) -> List[Tuple[torch.Tensor, float]]:
        """
        Run Batched MCTS for a list of active games.

        Args:
            boards: List of current board states (one per game).

        Returns:
            List of (policy, root_value) tuples, one for each game.
        """
        # Create a fresh search tree for each game
        roots = [MCTSNode(board=b.copy()) for b in boards]

        # Initial expansion for all roots
        self._expand_nodes(roots)

        for _ in range(self.num_simulations):
            leaves = []
            paths = []  # Store the path taken for each game

            # 1. Selection
            for root in roots:
                node = root
                path = [node]

                # Traverse down until we hit a leaf or unexpanded node
                while node.is_expanded and node.children:
                    node = self._select_child(node)
                    path.append(node)

                leaves.append(node)
                paths.append(path)

            # 2. Expansion & Evaluation
            # We need to filter nodes that are already terminal or expanded
            nodes_to_expand = []
            indices_to_expand = []
            leaf_values = {}  # Map index -> value

            for i, leaf in enumerate(leaves):
                if leaf.board.is_game_over():
                    # Game over - terminal value
                    leaf_values[i] = self._terminal_value(leaf.board)
                elif not leaf.is_expanded:
                    # Needs expansion
                    nodes_to_expand.append(leaf)
                    indices_to_expand.append(i)
                else:
                    # Already expanded but has no children
                    leaf_values[i] = 0.0

            # Batch expansion for all valid leaves
            if nodes_to_expand:
                values = self._expand_nodes(nodes_to_expand)
                for idx, val in zip(indices_to_expand, values):
                    leaf_values[idx] = val

            # 3. Backpropagation
            for i, path in enumerate(paths):
                value = leaf_values[i]
                self._backpropagate(path, value)

        # Extract Policies
        results = []
        for root in roots:
            results.append((self._get_policy(root), root.value))

        return results

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score and ensure its board is generated."""
        best_child = max(
            node.children.values(),
            key=lambda child: child.ucb_score(self.c_puct, node.visit_count)
        )

        # Lazy Board Generation
        if best_child.board is None:
            assert node.board is not None, "Parent board cannot be None"
            assert best_child.move is not None, "Child move cannot be None"

            best_child.board = node.board.copy()
            best_child.board.push(best_child.move)

        return best_child

    def _expand_nodes(self, nodes: List[MCTSNode]) -> List[float]:
        """
        Expand a batch of nodes.
        Runs NN inference once for the whole batch.

        Returns:
            List of value estimates for each node.
        """
        if not nodes:
            return []

        # Batch encode
        # Note: Optimization possibility -> batch encode in StateEncoder
        # but for now list comprehension is robust.
        obs_list = [self.encoder.encode_node(node) for node in nodes]
        obs_tensor = torch.stack(obs_list).to(self.device)

        # Batch Inference
        # Optional: Mixed Precision could be handled here if strictly needed,
        # but usually handled in trainer or by pure performance of fp32 on inference.
        # We'll rely on the model's forward
        with torch.no_grad():
            logits, values = self.model(obs_tensor)

        # Convert to CPU for tree building
        policy_probs = torch.softmax(logits, dim=1).cpu().numpy()
        values = values.cpu().numpy().flatten()

        # Process each node
        for i, node in enumerate(nodes):
            assert node.board is not None, "Node to expand must have a board"
            board = node.board
            probs = policy_probs[i]
            apply_dirichlet = (
                node.parent is None
                and self.dirichlet_epsilon > 0.0
                and self.dirichlet_alpha > 0.0
            )

            if apply_dirichlet:
                legal_moves = [
                    move for move in board.legal_moves
                    if not move.promotion or move.promotion == chess.QUEEN
                ]
                if legal_moves:
                    noise = np.random.dirichlet(
                        [self.dirichlet_alpha] * len(legal_moves)
                    )
                else:
                    noise = None

            # Mask illegal moves and re-normalize (optional but good for exploration)
            # Actually AlphaZero relies on the network learning legal moves,
            # but masking is safer for the tree search.

            noise_idx = 0
            for move in board.legal_moves:
                # AlphaZero standard: Queen promotion only (simplification)
                if move.promotion and move.promotion != chess.QUEEN:
                     continue

                action_idx = move.from_square * 64 + move.to_square
                prior = probs[action_idx]
                if apply_dirichlet and noise is not None:
                    # Root-only exploration noise (AlphaZero-style).
                    prior = (1.0 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * noise[noise_idx]
                    noise_idx += 1

                # Create child WITHOUT copying board (Lazy)
                child = MCTSNode(
                    board=None,  # Will be created on visit
                    parent=node,
                    move=move,
                    prior=float(prior)
                )
                node.children[action_idx] = child

            node.is_expanded = True

        return values.tolist()

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Backpropagate value up the path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective

    def _terminal_value(self, board: chess.Board) -> float:
        """Get value for terminal position."""
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _get_policy(self, root: MCTSNode) -> torch.Tensor:
        """Convert visit counts to policy distribution."""
        policy = torch.zeros(4096, dtype=torch.float32, device=self.device)

        if not root.children:
            return policy

        # Collect visit counts
        for action_idx, child in root.children.items():
            policy[action_idx] = child.visit_count

        # Normalize
        total = policy.sum()
        if total > 0:
            if self.temperature == 0:
                # Deterministic (argmax) - usually for eval
                best_idx = policy.argmax()
                policy.zero_()
                policy[best_idx] = 1.0
            else:
                # Temperature scaling
                if self.temperature != 1.0:
                    policy = policy ** (1.0 / self.temperature)
                    total = policy.sum()
                policy = policy / total

        return policy

    def select_move(self, board: chess.Board) -> Tuple[chess.Move, torch.Tensor, float]:
        """Legacy helper for single-game compatibility (optional tests)."""
        results = self.search_batch([board])
        policy, value = results[0]

        # Sample action
        if self.temperature > 0:
            action_idx = int(torch.multinomial(policy, 1).item())
        else:
            action_idx = int(policy.argmax().item())

        from_sq = action_idx // 64
        to_sq = action_idx % 64
        move = chess.Move(from_sq, to_sq)

        if chess.square_rank(to_sq) in [0, 7]:
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                move.promotion = chess.QUEEN

        return move, policy, value
