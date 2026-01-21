#include <vector>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <random>

#include "chess.hpp"

namespace mcts {

struct Node {
    chess::Board board;
    Node* parent = nullptr;
    chess::Move move = chess::Move::NULL_MOVE;

    int visit_count = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    bool is_expanded = false;

    std::vector<std::unique_ptr<Node>> children;

    Node(const chess::Board& b, Node* p, chess::Move m, float pr)
        : board(b), parent(p), move(m), prior(pr) {
            children.reserve(32); // Pre-allocate for typical branching factor
        }

    // Disable copying
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    float value() const {
        return visit_count > 0 ? value_sum / visit_count : 0.0f;
    }

    float ucb_score(float c_puct, int parent_visits) const {
        if (visit_count == 0) return 10000000.0f; // Infinity
        float q_val = value();
        float u_val = c_puct * prior * std::sqrt((float)parent_visits) / (1.0f + visit_count);
        return q_val + u_val;
    }
};

class MCTS {
public:
    MCTS(float c_puct, int num_simulations)
        : c_puct_(c_puct), num_simulations_(num_simulations) {}

    void reset(const std::string& fen) {
        root_ = std::make_unique<Node>(chess::Board(fen), nullptr, chess::Move::NULL_MOVE, 0.0f);
    }

    Node* select_leaf() {
        Node* node = root_.get();
        while (node->is_expanded && !node->children.empty()) {
            Node* best_child = nullptr;
            float best_score = -1e9;

            for (const auto& child : node->children) {
                float score = child->ucb_score(c_puct_, node->visit_count);
                if (score > best_score) {
                    best_score = score;
                    best_child = child.get();
                }
            }
            node = best_child;
        }
        return node;
    }

    // New optimized batch selection
    static void select_leaf_batch(
        std::vector<MCTS*>& trees,
        std::vector<Node*>& out_leaves
    ) {
        int batch_size = trees.size();
        out_leaves.resize(batch_size);

        #pragma omp parallel for if(batch_size > 16)
        for(int i=0; i<batch_size; ++i) {
            out_leaves[i] = trees[i]->select_leaf();
        }
    }

    // Called by Python after NN inference
    // policy_probs: dense vector of 4096 floats.
    void expand(Node* node, const std::vector<float>& policy_probs, float value) {
        if (node->is_expanded) return;

        chess::Board& board = node->board;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        for (const auto& move : moves) {
            int from = move.from().index();
            int to = move.to().index();
            int action_idx = from * 64 + to;

            float prior = 0.0f;
            if (action_idx < (int)policy_probs.size()) {
                prior = policy_probs[action_idx];
            }

            // Create child
            chess::Board child_board = board;
            child_board.makeMove(move);

            node->children.push_back(std::make_unique<Node>(
                child_board, node, move, prior
            ));
        }

        node->is_expanded = true;
        backpropagate(node, value);
    }

    // New optimized batch expand to avoid Python loop
    static void expand_batch(
        const std::vector<Node*>& leaves,
        const float* policy_data, // Flat array [batch, 4096]
        const float* values,      // Flat array [batch]
        int batch_size
    ) {
        // Parallelize? It's O(batch_size * legal_moves).
        // 256 games. Fast enough single thread, but omp helps.
        #pragma omp parallel for if(batch_size > 16)
        for(int i=0; i<batch_size; ++i) {
            Node* leaf = leaves[i];
            float val = values[i];

            // Pointer to this leaf's policy section
            // policy_data is [batch_idx * 4096 + action_idx]
            // We need to know which policy row corresponds to this leaf?
            // "leaves" corresponds to "values" and "policy_data" index-wise.
            const float* leaf_policy = policy_data + (i * 4096);

            if (leaf && !leaf->is_expanded) {
                chess::Board& board = leaf->board;
                chess::Movelist moves;
                chess::movegen::legalmoves(moves, board);

                // Reserve memory if possible?
                // leaf->children.reserve(moves.size());

                for (const auto& move : moves) {
                    int from = move.from().index();
                    int to = move.to().index();
                    int action_idx = from * 64 + to;

                    float prior = 0.0f;
                    if (action_idx < 4096) {
                        prior = leaf_policy[action_idx];
                    }

                    // Create child
                    chess::Board child_board = board;
                    child_board.makeMove(move);

                    // Add child (thread safe? No, leaf is distinct per thread i)
                    leaf->children.push_back(std::make_unique<Node>(
                        child_board, leaf, move, prior
                    ));
                }
                leaf->is_expanded = true;

                // Backprop (thread safe? NO if multiple leaves share parents/root?)
                // BUT: In batched MCTS, each game is a separate tree.
                // leaves[i] belongs to tree[i].
                // So concurrent backprops are disjoint. Safe.
                // UNLESS we are sharing trees?
                // Currently mcts_cpp.py maintains separate MCTS objects.
                // So disjoint parents. Safe.

                Node* node = leaf;
                while (node != nullptr) {
                    node->visit_count++;
                    node->value_sum += val;
                    val = -val;
                    node = node->parent;
                }
            }
        }
    }

    void backpropagate(Node* node, float value) {
        while (node != nullptr) {
            node->visit_count++;
            node->value_sum += value;
            value = -value; // Flip perspective
            node = node->parent;
        }
    }

    std::string get_fen(Node* node) {
        return node->board.getFen();
    }

    int game_status(Node* node) {
         chess::Movelist moves;
         chess::movegen::legalmoves(moves, node->board);
         if (moves.empty()) {
             if (node->board.inCheck()) return 1; // Checkmate
             return 2; // Stalemate
         }
         return 0; // Active
    }

    std::vector<std::pair<int, int>> get_root_counts() {
        std::vector<std::pair<int, int>> counts;
        if (!root_) return counts;

        for (const auto& child : root_->children) {
            int from = child->move.from().index();
            int to = child->move.to().index();
            int idx = from * 64 + to;
            counts.push_back({idx, child->visit_count});
        }
        return counts;
    }

    float get_root_value() {
        return root_ ? root_->value() : 0.0f;
    }

    int get_root_visits() { return root_ ? root_->visit_count : 0; }

private:
    std::unique_ptr<Node> root_;
    float c_puct_;
    int num_simulations_;
};

inline void fill_plane(float* encoded_data, int batch_idx, int plane_idx, uint64_t bb, int perspective) {
    while (bb) {
        int sq = __builtin_ctzll(bb);
        int r = sq / 8;
        int c = sq % 8;

        if (perspective == 1) { // Black
            r = 7 - r;
            c = 7 - c;
        }

        // Shape: [Batch, 116, 8, 8] -> Flat index
        int idx = batch_idx * (116 * 64) + plane_idx * 64 + r * 8 + c;
        encoded_data[idx] = 1.0f;

        bb &= bb - 1;
    }
}

void encode_single_node(Node* node, float* data, int batch_idx) {
    if (!node) return;

    int perspective = (int)node->board.sideToMove(); // 0=White, 1=Black

    // 1. History (0-95)
    Node* current = node;
    for (int t = 0; t < 8; ++t) {
        if (!current) break;

        int base_plane = t * 12;
        const auto& board = current->board;

        // My pieces
        for (int pt = 0; pt < 6; ++pt) {
             uint64_t bb = board.pieces(static_cast<chess::PieceType::underlying>(pt), static_cast<chess::Color::underlying>(perspective)).getBits();
             fill_plane(data, batch_idx, base_plane + pt, bb, perspective);
        }

        // Enemy pieces
        for (int pt = 0; pt < 6; ++pt) {
             uint64_t bb = board.pieces(static_cast<chess::PieceType::underlying>(pt), static_cast<chess::Color::underlying>(1 - perspective)).getBits();
             fill_plane(data, batch_idx, base_plane + 6 + pt, bb, perspective);
        }

        current = current->parent;
    }

    // 2. Metadata
    int meta_base = 96;
    const auto& b = node->board;

    // En Passant (100)
    if (b.enpassantSq() != chess::Square::NO_SQ) {
        int sq = b.enpassantSq().index();
        int r = sq / 8;
        int c = sq % 8;
        if (perspective == 1) { r = 7 - r; c = 7 - c; }
        int idx = batch_idx * (116 * 64) + (meta_base + 4) * 64 + r * 8 + c;
        data[idx] = 1.0f;
    }

    // Halfmove (101)
    float halfmove = std::min(b.halfMoveClock() / 50.0f, 1.0f);
    int p_idx = meta_base + 5;
    for(int i=0; i<64; ++i) data[batch_idx*(116*64) + p_idx*64 + i] = halfmove;

    // Fullmove (102)
    float fullmove = std::min(b.fullMoveNumber() / 200.0f, 1.0f);
    p_idx = meta_base + 6;
    for(int i=0; i<64; ++i) data[batch_idx*(116*64) + p_idx*64 + i] = fullmove;

    // Color (103)
    if (perspective == 1) { // Black
        p_idx = meta_base + 7;
        for(int i=0; i<64; ++i) data[batch_idx*(116*64) + p_idx*64 + i] = 1.0f;
    }
}

} // namespace mcts
