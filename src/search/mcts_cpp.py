import torch
import numpy as np
import chess
import chess.engine
from typing import List, Tuple, Optional, Any
import mcts_cpp
from src.core.config import AppConfig


class MCTS:
    """
    Batched MCTS using High-Performance C++ Backend.
    Supports virtual-loss batched leaf evaluation and tree reuse.
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
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 3,
        reuse_tree: bool = True,
        leaves_per_sim: int = 8,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.reuse_tree = reuse_tree
        self.leaves_per_sim = leaves_per_sim

        self.trees: List[Optional[mcts_cpp.MCTS]] = []

        # Optional: Direct synchronous Stockfish for tree search
        self._stockfish_engine = None
        self._stockfish_depth = stockfish_depth
        if stockfish_path:
            try:
                self._stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self._stockfish_engine.configure({'Threads': 1, 'Hash': 16})
                print(f'MCTS: Direct Stockfish engine initialized from {stockfish_path}')
            except Exception as e:
                print(f'MCTS: Failed to init Stockfish: {e}')
                self._stockfish_engine = None

    def _evaluate_fens_sync(self, fens: List[str]) -> List[float]:
        """Synchronously evaluate FENs with internal Stockfish engine."""
        results = []
        for fen in fens:
            try:
                board = chess.Board(fen)
                info = self._stockfish_engine.analyse(board, chess.engine.Limit(depth=self._stockfish_depth))
                score_obj = info.get('score')
                if score_obj is None:
                    results.append(0.0)
                    continue

                relative_score = score_obj.relative
                if relative_score.is_mate():
                    mate_moves = relative_score.mate()
                    score = 10000 if (mate_moves and mate_moves > 0) else -10000
                else:
                    val = relative_score.score()
                    score = val if val is not None else 0

                results.append(float(np.tanh(-score / 100.0)))
            except Exception:
                results.append(0.0)
        return results

    def search_batch(
        self,
        boards: List[chess.Board],
        verifier: Optional[Any] = None,
        last_moves: Optional[List[Optional[chess.Move]]] = None,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Run Batched MCTS using C++ with virtual-loss batched leaf evaluation.

        Args:
            boards: List of current board states.
            verifier: Optional AsyncStockfishVerifier for hybrid search.
            last_moves: Optional list of last moves played per game slot (for tree reuse).
        """
        batch_size = len(boards)

        # Ensure we have enough C++ trees
        while len(self.trees) < batch_size:
            self.trees.append(
                mcts_cpp.MCTS(
                    self.c_puct,
                    self.num_simulations,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                )
            )

        # 1. Reset / Reuse logic
        active_trees = []
        for i, board in enumerate(boards):
            tree = self.trees[i]
            reused = False
            if self.reuse_tree and last_moves and last_moves[i] is not None:
                move = last_moves[i]
                from_sq = move.from_square
                to_sq = move.to_square
                reused = tree.advance_root(from_sq, to_sq)
            if not reused:
                tree.reset(board.fen())
            active_trees.append(tree)

        # 2. Run Simulations with virtual-loss batched leaf selection
        k = self.leaves_per_sim
        sims_left = self.num_simulations

        while sims_left > 0:
            step_k = k if sims_left >= k else sims_left
            sims_left -= step_k

            # Select K leaves per tree (with virtual loss)
            all_leaves, tree_indices = mcts_cpp.select_leaves_batch_vl(
                active_trees, step_k
            )

            if not all_leaves:
                break

            # Categorize into expand vs verify
            expand_indices = []
            verify_indices = []

            for i, leaf in enumerate(all_leaves):
                if leaf.is_expanded:
                    if verifier:
                        verify_indices.append(i)
                else:
                    expand_indices.append(i)
                    if leaf.depth == 1 and verifier:
                        verify_indices.append(i)

            # Stockfish verification
            stockfish_values = {}
            use_internal_sf = self._stockfish_engine is not None
            use_verifier = verifier is not None and not use_internal_sf

            if verify_indices and (use_internal_sf or use_verifier):
                fens = []
                for idx in verify_indices:
                    tree = active_trees[tree_indices[idx]]
                    leaf = all_leaves[idx]
                    fens.append(tree.get_fen(leaf))

                if use_internal_sf:
                    sf_results = self._evaluate_fens_sync(fens)
                else:
                    sf_results = verifier.verify_batch(fens)

                for idx, val in zip(verify_indices, sf_results):
                    stockfish_values[idx] = val

            # NN evaluation for unexpanded leaves
            if expand_indices:
                subset_leaves = [all_leaves[i] for i in expand_indices]
                subset_trees = [active_trees[tree_indices[i]] for i in expand_indices]

                encoded_states = mcts_cpp.encode_batch(subset_leaves)
                obs_tensor = torch.from_numpy(encoded_states).to(self.device).float()

                with torch.no_grad():
                    logits, values = self.model(obs_tensor)

                policy_probs = torch.softmax(logits, dim=1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()

                policy_probs = np.ascontiguousarray(policy_probs, dtype=np.float32)
                values_np = np.ascontiguousarray(values_np, dtype=np.float32)

                # Patch values with Stockfish if available
                for j, leaf_idx in enumerate(expand_indices):
                    if leaf_idx in stockfish_values:
                        values_np[j] = stockfish_values[leaf_idx]

                # expand_batch_fast handles expansion + backprop (with VL undo)
                mcts_cpp.expand_batch_fast(subset_trees, subset_leaves, policy_probs, values_np)

            # Handle hijack updates (already expanded, just need VL undo + value update)
            for idx in verify_indices:
                leaf = all_leaves[idx]
                if leaf.is_expanded and idx not in expand_indices:
                    if idx in stockfish_values:
                        active_trees[tree_indices[idx]].undo_virtual_loss_and_update(
                            leaf, stockfish_values[idx]
                        )

            # If no verifier is used, undo virtual loss for expanded leaves
            if verifier is None:
                for i, leaf in enumerate(all_leaves):
                    if leaf.is_expanded and i not in expand_indices:
                        active_trees[tree_indices[i]].undo_virtual_loss(leaf)

        # 3. Extract Results
        results = []
        for tree in active_trees:
            counts = tree.get_root_counts()
            policy = torch.zeros(4096, dtype=torch.float32, device=self.device)
            for idx, count in counts:
                policy[idx] = count

            total = policy.sum()
            if total > 0:
                policy = policy / total
                if self.temperature != 1.0 and self.temperature > 0:
                    policy = policy ** (1.0 / self.temperature)
                    policy = policy / policy.sum()

            value = tree.get_root_value()
            results.append((policy, value))

        return results
