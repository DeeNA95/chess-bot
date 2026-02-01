
import math
import chess
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import concurrent.futures

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

    # Hybrid MCTS fields
    verified: bool = False

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


class MCTSHybrid:
    """
    Batched Monte Carlo Tree Search with Stockfish Verification (Hybrid).
    Pure Python implementation.
    """

    def __init__(
        self,
        model,
        encoder,
        verifier,  # AsyncStockfishVerifier
        device: str = 'cpu',
        num_simulations: int = 50,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        self.model = model
        self.encoder = encoder
        self.verifier = verifier
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    def search_batch(self, boards: List[chess.Board]) -> List[Tuple[torch.Tensor, float]]:
        """
        Run Batched MCTS for a list of active games.
        """
        # Create a fresh search tree for each game
        roots = [MCTSNode(board=b.copy()) for b in boards]

        # Initial expansion for all roots
        self._expand_nodes(roots)

        for _ in range(self.num_simulations):
            leaves = []
            paths = []

            # Nodes that need verification this iteration
            nodes_to_verify = []
            indices_to_verify = []

            # 1. Selection
            for i, root in enumerate(roots):
                node = root
                path = [node]

                # Traverse down
                while node.is_expanded and node.children:
                    # Check for hijack: Re-verify at 5 visits
                    # Note: In C++, we check visit_count >= 5.
                    # Here we check before selecting child.
                    # If current node has >= 5 visits and NOT verified, hijack it.
                    # Exception: Root usually not hijacked unless desired.
                    # The C++ conditions: depth > 0 && visit_count >= 5 && !verified.
                    if node.parent is not None and node.visit_count >= 5 and not node.verified:
                        # Hijack!
                        nodes_to_verify.append(node)
                        indices_to_verify.append(len(leaves)) # Map leaf index back to something?
                        # Wait, we need to track this node as the leaf for this simulation.
                        break

                    node = self._select_child(node)
                    path.append(node)

                leaves.append(node)
                paths.append(path)

            # 2. Verification
            # Filter leaves that were hijacked
            hijacked_leaves = []
            hijacked_indices_in_leaves = []

            for i, leaf in enumerate(leaves):
                # Check if it was a hijack stop (already expanded/has children but stopped)
                # Or simply reached a leaf that meets criteria?
                # The 'break' above stops traversal.
                # If 'node' has children and we stopped, it's a hijack.
                if leaf.is_expanded and leaf.children and not leaf.verified and leaf.visit_count >= 5 and leaf.parent is not None:
                     hijacked_leaves.append(leaf)
                     hijacked_indices_in_leaves.append(i)

            if hijacked_leaves:
                fens = [leaf.board.fen() for leaf in hijacked_leaves]
                # print(f"DEBUG: Verifying {len(fens)} positions...")
                sf_values = self.verifier.verify_batch(fens)

                for leaf, val in zip(hijacked_leaves, sf_values):
                    leaf.verified = True
                    # Update value logic:
                    # Standard MCTS updates value sum.
                    # Hijack replaces/mixes value?
                    # "Update value" usually means we discard the subtree params or just backprop this new value?
                    # In C++: update_value(leaf, val) adds val to value_sum and increments visit_count.
                    # And it backprops.
                    # But since we stopped selection at this node, we treat it as the leaf for this sim.
                    # So we will backpropagate 'val' from here.
                    self._backpropagate([n for n in paths[leaves.index(leaf)]], val)

            # 3. Expansion & Evaluation (for non-hijacked)
            nodes_to_expand = []
            expand_indices_in_leaves = []
            leaf_values_for_backprop = {} # idx -> value

            for i, leaf in enumerate(leaves):
                # If hijacked, already backpropped. Skip.
                if i in hijacked_indices_in_leaves:
                    continue

                if leaf.board.is_game_over():
                    val = self._terminal_value(leaf.board)
                    self._backpropagate(paths[i], val)
                elif not leaf.is_expanded:
                    nodes_to_expand.append(leaf)
                    expand_indices_in_leaves.append(i)
                else:
                    # Expanded, not hijacked, but ended up here?
                    # Should not happen unless select_child failed (no children).
                    # Treat as 0 or reuse value.
                    self._backpropagate(paths[i], 0.0)

            if nodes_to_expand:
                values = self._expand_nodes(nodes_to_expand)
                for idx, val in zip(expand_indices_in_leaves, values):
                    self._backpropagate(paths[idx], val)

        # Extract Policies
        results = []
        for root in roots:
            results.append((self._get_policy(root), root.value))

        return results

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB."""
        best_child = max(
            node.children.values(),
            key=lambda child: child.ucb_score(self.c_puct, node.visit_count)
        )
        if best_child.board is None:
            best_child.board = node.board.copy()
            best_child.board.push(best_child.move)
        return best_child

    def _expand_nodes(self, nodes: List[MCTSNode]) -> List[float]:
        """Expand batch of nodes using model."""
        if not nodes:
            return []

        obs_list = [self.encoder.encode_node(node) for node in nodes]
        obs_tensor = torch.stack(obs_list).to(self.device).float()

        with torch.no_grad():
            logits, values = self.model(obs_tensor)

        policy_probs = torch.softmax(logits, dim=1).cpu().numpy()
        values = values.cpu().numpy().flatten()

        for i, node in enumerate(nodes):
            board = node.board
            probs = policy_probs[i]

            for move in board.legal_moves:
                if move.promotion and move.promotion != chess.QUEEN:
                    continue

                action_idx = move.from_square * 64 + move.to_square
                prior = probs[action_idx]

                child = MCTSNode(
                    board=None,
                    parent=node,
                    move=move,
                    prior=float(prior)
                )
                node.children[action_idx] = child

            node.is_expanded = True

        return values.tolist()

    def _backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _get_policy(self, root: MCTSNode) -> torch.Tensor:
        policy = torch.zeros(4096, dtype=torch.float32, device=self.device)
        if not root.children:
            return policy
        for action_idx, child in root.children.items():
            policy[action_idx] = child.visit_count

        total = policy.sum()
        if total > 0:
            if self.temperature == 0:
                best_idx = policy.argmax()
                policy.zero_()
                policy[best_idx] = 1.0
            else:
                if self.temperature != 1.0:
                    policy = policy ** (1.0 / self.temperature)
                    total = policy.sum()
                policy = policy / total
        return policy
