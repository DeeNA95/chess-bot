import torch
import numpy as np
import chess
from typing import List, Tuple, Optional
from src import mcts_cpp
from src.core.config import AppConfig

class MCTS:
    """
    Batched MCTS using High-Performance C++ Backend.
    """
    def __init__(
        self,
        model,
        encoder,
        device: str = 'cpu',
        num_simulations: int = 50,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        # Cache for C++ trees: map active game index (or just parallel slot) to C++ object
        self.trees: List[Optional[mcts_cpp.MCTS]] = []

    def search_batch(self, boards: List[chess.Board]) -> List[Tuple[torch.Tensor, float]]:
        """
        Run Batched MCTS using C++.
        Args:
            boards: List of current board states.
                    Assumes the order corresponds to stable game slots.
        """
        batch_size = len(boards)

        # Ensure we have enough C++ trees
        while len(self.trees) < batch_size:
            self.trees.append(mcts_cpp.MCTS(self.c_puct, self.num_simulations))

        # 1. Sync / Reset logic
        active_trees = []
        for i, board in enumerate(boards):
            tree = self.trees[i]
            # Simple V1: Always reset to current board FEN.
            tree.reset(board.fen())
            active_trees.append(tree)

        # 2. Run Simulations
        for _ in range(self.num_simulations):
            leaves = []
            leaf_indices = []

            # Selection
            # Selection
            # FAST PATH: C++ Batch Selection
            leaves = mcts_cpp.select_leaf_batch(active_trees)
            if not leaves:
                break

            # leaf_indices redundant if we process all active_trees in order.
            # active_trees[i] corresponds to leaves[i].
            # expand_batch_fast expects leaves and policy in same order.
            # So policy_probs[i] (from model on leaves[i]) matches leaves[i].
            # We don't need leaf_indices anymore for backprop because we batch backprop too.
            # And we don't need to rebuild leaves list manually.

            if not leaves:
                break

            # Expansion (Batch Inference)
            # Use C++ Encoder (returns numpy array [B, 116, 8, 8])
            encoded_states = mcts_cpp.encode_batch(leaves)
            obs_tensor = torch.from_numpy(encoded_states).to(self.device).float()

            with torch.no_grad():
                logits, values = self.model(obs_tensor)

            # Backprop
            policy_probs = torch.softmax(logits, dim=1).cpu().numpy()
            values_np = values.cpu().numpy().flatten()

            # FAST PATH: C++ Batch Expansion
            # We pass the full arrays. C++ slices them based on leaf index.
            # leaves list corresponds 1:1 to rows in policy_probs/values_np?
            # Yes, 'leaves' list was built from 'active_trees'.
            # 'leaf_indices' maps leaf[j] -> tree index.
            # Wait, 'leaves' in expansion corresponds to the batch we just forwarded.
            # So policy_probs[j] corresponds to leaves[j].
            # C++ expand_batch takes (leaves, policy, values).
            # It assumes policy[i] corresponds to leaves[i].

            # Cast leaves to list of Node* (already are)
            # Ensure types for pybind
            policy_probs = np.ascontiguousarray(policy_probs, dtype=np.float32)
            values_np = np.ascontiguousarray(values_np, dtype=np.float32)

            mcts_cpp.expand_batch_fast(leaves, policy_probs, values_np)

        # 3. Extract Results
        results = []
        for tree in active_trees:
            # Policy
            counts = tree.get_root_counts() # list of (idx, count)
            policy = torch.zeros(4096, dtype=torch.float32, device=self.device)
            for idx, count in counts:
                policy[idx] = count

            # Normalize
            total = policy.sum()
            if total > 0:
                policy = policy / total
                if self.temperature != 1.0 and self.temperature > 0:
                     policy = policy ** (1.0 / self.temperature)
                     policy = policy / policy.sum()

            # Value
            value = tree.get_root_value()
            results.append((policy, value))

        return results
