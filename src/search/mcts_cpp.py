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
    """
    def __init__(
        self,
        model,
        encoder,
        device: str = 'cpu',
        num_simulations: int = 50,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        stockfish_path: Optional[str] = None,  # Direct Stockfish for tree search
        stockfish_depth: int = 3,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        # Cache for C++ trees: map active game index (or just parallel slot) to C++ object
        self.trees: List[Optional[mcts_cpp.MCTS]] = []

        # Optional: Direct synchronous Stockfish for tree search (no multiprocessing)
        self._stockfish_engine = None
        self._stockfish_depth = stockfish_depth
        if stockfish_path:
            try:
                self._stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self._stockfish_engine.configure({'Threads': 1, 'Hash': 16})
                print(f'🔧 MCTS: Direct Stockfish engine initialized from {stockfish_path}')
            except Exception as e:
                print(f'⚠️ MCTS: Failed to init Stockfish: {e}')
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

                # Negate because relative_score is from opponent's perspective after move
                results.append(float(np.tanh(-score / 100.0)))
            except Exception as e:
                results.append(0.0)
        return results

    def search_batch(self, boards: List[chess.Board], verifier: Optional[Any] = None) -> List[Tuple[torch.Tensor, float]]:
        """
        Run Batched MCTS using C++.
        Args:
            boards: List of current board states.
            verifier: Optional AsyncStockfishVerifier (or similar) for hybrid search.
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
            # 1. Batched Selection
            # print("DEBUG: Calling select_leaf_batch...", flush=True)
            leaves = mcts_cpp.select_leaf_batch(active_trees)
            # print(f"DEBUG: select_leaf_batch done. Got {len(leaves)} leaves.", flush=True)
            if not leaves:
                break

            # 2. Categorize Leaves
            # We have two types of work:
            # A. Expansion (NN) - for unexpanded nodes
            # B. Verification (Stockfish) - for Depth 1 override OR High-Visit hijack

            expand_indices = []
            verify_indices = []

            for i, leaf in enumerate(leaves):
                if leaf.is_expanded:
                    # Hijacked node (High Visit Count) -> Needs Verification
                    if verifier:
                        verify_indices.append(i)
                else:
                    # Unexpanded Leaf -> Needs Expansion (NN)
                    expand_indices.append(i)
                    # If Depth 1 (Root Child) and we have verifier, we ALSO need Verification
                    if leaf.depth == 1 and verifier:
                        verify_indices.append(i)

            # 3. Validation / Backfill values
            stockfish_values = {} # Map leaf_index -> value

            # Use internal Stockfish engine if available, else use passed verifier
            use_internal_sf = self._stockfish_engine is not None
            use_verifier = verifier is not None and not use_internal_sf

            if verify_indices and (use_internal_sf or use_verifier):
                # Collect FENs
                fens = []
                for idx in verify_indices:
                    tree = active_trees[idx]
                    leaf = leaves[idx]
                    fens.append(tree.get_fen(leaf))

                # Evaluate with Stockfish
                if use_internal_sf:
                    sf_results = self._evaluate_fens_sync(fens)
                else:
                    sf_results = verifier.verify_batch(fens)

                for idx, val in zip(verify_indices, sf_results):
                    stockfish_values[idx] = val

            # 4. Expansion (NN)
            if expand_indices:
                subset_leaves = [leaves[i] for i in expand_indices]
                subset_trees = [active_trees[i] for i in expand_indices]  # Trees for pool allocation
                # print(f"DEBUG: Encoding {len(subset_leaves)} states...", flush=True)
                encoded_states = mcts_cpp.encode_batch(subset_leaves)
                # print(f"DEBUG: Encoding done. Shape: {encoded_states.shape}", flush=True)
                obs_tensor = torch.from_numpy(encoded_states).to(self.device).float()

                with torch.no_grad():
                    # print("DEBUG: Run Model Forward...", flush=True)
                    logits, values = self.model(obs_tensor)
                    # print("DEBUG: Model Forward done.", flush=True)

                policy_probs = torch.softmax(logits, dim=1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()

                policy_probs = np.ascontiguousarray(policy_probs, dtype=np.float32)
                values_np = np.ascontiguousarray(values_np, dtype=np.float32)

                # Patch values with Stockfish if available
                patch_indices = []
                patch_values = []

                # We need to construct the arrays expected by expand_batch_fast
                # But expand_batch_fast takes the subset arrays.
                # iterate over subset
                for k, leaf_idx in enumerate(expand_indices):
                    if leaf_idx in stockfish_values:
                        values_np[k] = stockfish_values[leaf_idx]

                mcts_cpp.expand_batch_fast(subset_trees, subset_leaves, policy_probs, values_np)
                # print("DEBUG: expand_batch_fast done.", flush=True)

            # 5. Pure Updates (High Visit Hijacks)
            # These are nodes that were ALREADY expanded, but we just got a new value for them.
            # We do NOT request expansion. We just update.
            # However verify_indices includes depth 1 leaves too. We filtered those out in step 4 check?
            # No, we handled them in step 4 by patching.
            # We typically only 'update' nodes that represent pure hijacks (is_expanded=True).

            for idx in verify_indices:
                leaf = leaves[idx]
                if leaf.is_expanded and idx not in expand_indices:
                     # This is a pure update (High Visit Hijack)
                     if idx in stockfish_values:
                         active_trees[idx].update_value(leaf, stockfish_values[idx])

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
