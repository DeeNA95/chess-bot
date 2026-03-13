#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <mutex>
// #include <numeric>
#include <random>

#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "chess.hpp"

// ===== AlphaZero Action Encoding (4672 = 73 move types × 64 squares) =====
static constexpr int ACTION_SPACE_SIZE = 4672;
static constexpr int NUM_MOVE_TYPES = 73;
static constexpr int NUM_SQUARES_AZ = 64;

// 8 compass directions: N, NE, E, SE, S, SW, W, NW
static const int AZ_DIRECTIONS[8][2] = {
    {1, 0}, {1, 1}, {0, 1}, {-1, 1},
    {-1, 0}, {-1, -1}, {0, -1}, {1, -1},
};

// 8 knight moves
static const int AZ_KNIGHT_MOVES[8][2] = {
    {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
    {1, 2}, {1, -2}, {-1, 2}, {-1, -2},
};

// Underpromotion directions (delta_file)
static const int AZ_UNDERPROMO_DIRS[3] = {-1, 0, 1};

static inline int az_flip_square(int sq) {
    int r = sq / 8, f = sq % 8;
    return (7 - r) * 8 + (7 - f);
}

// Encode a chess::Move into an AlphaZero action index [0, 4672)
static int move_to_action_idx(const chess::Move& move, chess::Color perspective) {
    int from_sq = move.from().index();
    int to_sq = move.to().index();

    if (perspective == chess::Color::BLACK) {
        from_sq = az_flip_square(from_sq);
        to_sq = az_flip_square(to_sq);
    }

    int from_rank = from_sq / 8, from_file = from_sq % 8;
    int to_rank = to_sq / 8, to_file = to_sq % 8;
    int dr = to_rank - from_rank;
    int df = to_file - from_file;

    // Underpromotions (knight, bishop, rook)
    if (move.typeOf() == chess::Move::PROMOTION) {
        auto pt = move.promotionType();
        if (pt != chess::PieceType::underlying::QUEEN) {
            int piece_idx = -1;
            if (pt == chess::PieceType::underlying::KNIGHT) piece_idx = 0;
            else if (pt == chess::PieceType::underlying::BISHOP) piece_idx = 1;
            else if (pt == chess::PieceType::underlying::ROOK) piece_idx = 2;

            int dir_idx = df + 1; // -1->0, 0->1, 1->2
            if (piece_idx >= 0 && dir_idx >= 0 && dir_idx < 3) {
                int move_type = 64 + piece_idx * 3 + dir_idx;
                return move_type * NUM_SQUARES_AZ + from_sq;
            }
        }
    }

    // Knight moves
    for (int i = 0; i < 8; ++i) {
        if (AZ_KNIGHT_MOVES[i][0] == dr && AZ_KNIGHT_MOVES[i][1] == df) {
            return (56 + i) * NUM_SQUARES_AZ + from_sq;
        }
    }

    // Queen-type moves (includes queen promotions)
    int norm_dr = (dr > 0) ? 1 : (dr < 0) ? -1 : 0;
    int norm_df = (df > 0) ? 1 : (df < 0) ? -1 : 0;
    int distance = std::max(std::abs(dr), std::abs(df));

    for (int i = 0; i < 8; ++i) {
        if (AZ_DIRECTIONS[i][0] == norm_dr && AZ_DIRECTIONS[i][1] == norm_df) {
            int move_type = i * 7 + (distance - 1);
            return move_type * NUM_SQUARES_AZ + from_sq;
        }
    }

    return 0; // fallback
}
// ===== End AlphaZero Action Encoding =====

namespace mcts {

// ============================================================================
// Node - MCTS tree node (uses raw pointers, pool owns memory)
// Defined FIRST so that NodePool can use std::deque<Node>
// ============================================================================
struct Node {
    chess::Board board;
    Node* parent = nullptr;
    chess::Move move = chess::Move::NULL_MOVE;

    int visit_count = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    bool is_expanded = false;
    int virtual_loss = 0;

    // Hybrid search fields
    int depth = 0;
    bool verified = false;
    uint32_t magic = 0x12345678;

    // Children are raw pointers - NodePool owns the memory
    std::vector<Node*> children;

    Node() = default;

    Node(const chess::Board& b, Node* p, chess::Move m, float pr, int d)
        : board(b), parent(p), move(m), prior(pr), depth(d) {
        children.reserve(35);  // Avg legal moves in chess
    }

    // Allow move operations
    Node(Node&&) = default;
    Node& operator=(Node&&) = default;

    // Disable copying
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    float value() const {
        int effective_visits = visit_count + virtual_loss;
        if (effective_visits == 0) return 0.0f;
        return (value_sum - (float)virtual_loss) / (float)effective_visits;
    }

    float ucb_score(float c_puct, int parent_visits) const {
        int effective_visits = visit_count + virtual_loss;
        if (effective_visits == 0) return 10000000.0f;
        float q_val = (value_sum - (float)virtual_loss) / (float)effective_visits;
        float u_val = c_puct * prior * std::sqrt((float)parent_visits) / (1.0f + effective_visits);
        return q_val + u_val;
    }
};

// ============================================================================
// NodePool - Thread-safe arena using std::deque for pointer stability
// std::deque guarantees that pointers/references remain valid when elements
// are added (unlike std::vector which may reallocate and invalidate all pointers)
// ============================================================================
class NodePool {
public:
    static constexpr size_t DEFAULT_CAPACITY = 100000;

    explicit NodePool(size_t capacity = DEFAULT_CAPACITY) : capacity_(capacity) {}

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Swap with an empty deque to release memory back to the OS.
        std::deque<Node> empty;
        nodes_.swap(empty);
    }

    // Thread-safe allocation of a new node from the pool
    Node* allocate(const chess::Board& b, Node* parent, chess::Move m, float pr, int d) {
        std::lock_guard<std::mutex> lock(mutex_);
        nodes_.emplace_back(b, parent, m, pr, d);
        return &nodes_.back();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_.size();
    }

    size_t capacity() const { return capacity_; }

private:
    std::deque<Node> nodes_;  // deque: pointer-stable on growth
    mutable std::mutex mutex_;
    size_t capacity_;
};

// ============================================================================
// MCTS - Monte Carlo Tree Search with NodePool
// ============================================================================
class MCTS {
public:
    MCTS(float c_puct, int num_simulations, float dirichlet_alpha, float dirichlet_epsilon)
        : c_puct_(c_puct),
          num_simulations_(num_simulations),
          dirichlet_alpha_(dirichlet_alpha),
          dirichlet_epsilon_(dirichlet_epsilon),
          pool_(NodePool::DEFAULT_CAPACITY),
          rng_(std::random_device{}()) {}

    void reset(const std::string& fen) {
        pool_.clear();
        root_ = pool_.allocate(chess::Board(fen), nullptr, chess::Move::NULL_MOVE, 0.0f, 0);
    }

    Node* select_leaf() {
        Node* node = root_;
        if (!node) {
            std::cerr << "FATAL: Root is null in select_leaf" << std::endl;
            std::abort();
        }
        if (node->magic != 0x12345678) {
            std::cerr << "FATAL: Root corrupted in select_leaf" << std::endl;
            std::abort();
        }

        while (node->is_expanded && !node->children.empty()) {
            if (node->magic != 0x12345678) {
                std::cerr << "FATAL: Node corrupted during traversal" << std::endl;
                std::abort();
            }

            // Check for hijack: Re-verify at 5 visits
            if (node->depth > 0 && node->visit_count >= 5 && !node->verified) {
                return node;
            }

            Node* best_child = nullptr;
            float best_score = -1e9;

            for (Node* child : node->children) {
                if (!child || child->magic != 0x12345678) {
                    std::cerr << "FATAL: Child corrupted during selection" << std::endl;
                    continue;
                }
                float score = child->ucb_score(c_puct_, node->visit_count);
                if (score > best_score) {
                    best_score = score;
                    best_child = child;
                }
            }
            if (!best_child) {
                return node;
            }
            node = best_child;
        }
        return node;
    }

    static void select_leaf_batch(
        std::vector<MCTS*>& trees,
        std::vector<Node*>& out_leaves
    ) {
        int batch_size = trees.size();
        out_leaves.resize(batch_size);

        // Sequential loop (OpenMP removed for stability with Python multiprocessing)
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < batch_size; ++i) {
            out_leaves[i] = trees[i]->select_leaf();
        }
    }

    void expand(Node* node, const std::vector<float>& policy_probs, float value) {
        if (!node || node->magic != 0x12345678) {
            std::cerr << "FATAL: Expand called on corrupted node" << std::endl;
            return;
        }
        if (node->is_expanded) return;

        chess::Board& board = node->board;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        const bool apply_dirichlet = (node->depth == 0 && dirichlet_epsilon_ > 0.0f && dirichlet_alpha_ > 0.0f);
        std::vector<float> noise;
        if (apply_dirichlet) {
            noise = sample_dirichlet(static_cast<int>(moves.size()), dirichlet_alpha_);
        }

        int move_idx = 0;
        for (const auto& move : moves) {
            int action_idx = move_to_action_idx(move, board.sideToMove());

            float prior = 0.0f;
            if (action_idx < (int)policy_probs.size()) {
                prior = policy_probs[action_idx];
            }
            if (apply_dirichlet) {
                // Root-only exploration noise (AlphaZero-style).
                prior = (1.0f - dirichlet_epsilon_) * prior + dirichlet_epsilon_ * noise[move_idx];
            }

            chess::Board child_board = board;
            child_board.makeMove(move);

            Node* child = pool_.allocate(child_board, node, move, prior, node->depth + 1);
            node->children.push_back(child);
            ++move_idx;
        }

        node->is_expanded = true;
        backpropagate(node, value);
    }

    void update_value(Node* node, float value) {
        if (!node || node->magic != 0x12345678) {
            std::cerr << "FATAL: update_value corrupted" << std::endl;
            return;
        }
        node->verified = true;
        backpropagate(node, value);
    }

    // Static expand_batch needs access to pool - we pass MCTS pointers
    static void expand_batch(
        std::vector<MCTS*>& trees,
        const std::vector<Node*>& leaves,
        const float* policy_data,
        const float* values,
        int batch_size
    ) {
        if (batch_size <= 0) return;

        // Sequential loop (OpenMP removed for stability with Python multiprocessing)
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < batch_size; ++i) {
            Node* leaf = leaves[i];
            float val = values[i];
            const float* leaf_policy = policy_data + (i * ACTION_SPACE_SIZE);
            MCTS* tree = trees[i];

            if (leaf && !leaf->is_expanded) {
                if (leaf->magic != 0x12345678) {
                    std::cerr << "FATAL: expand_batch encountered corrupted leaf at index " << i << std::endl;
                    continue;
                }

                chess::Board& board = leaf->board;
                chess::Movelist moves;
                chess::movegen::legalmoves(moves, board);

                const bool apply_dirichlet = (leaf->depth == 0 && tree->dirichlet_epsilon_ > 0.0f && tree->dirichlet_alpha_ > 0.0f);
                std::vector<float> noise;
                if (apply_dirichlet) {
                    noise = tree->sample_dirichlet(static_cast<int>(moves.size()), tree->dirichlet_alpha_);
                }

                int move_idx = 0;
                for (const auto& move : moves) {
                    int action_idx = move_to_action_idx(move, board.sideToMove());

                    float prior = 0.0f;
                    if (action_idx < ACTION_SPACE_SIZE) {
                        prior = leaf_policy[action_idx];
                    }
                    if (apply_dirichlet) {
                        // Root-only exploration noise (AlphaZero-style).
                        prior = (1.0f - tree->dirichlet_epsilon_) * prior + tree->dirichlet_epsilon_ * noise[move_idx];
                    }

                    chess::Board child_board = board;
                    child_board.makeMove(move);

                    Node* child = tree->pool_.allocate(child_board, leaf, move, prior, leaf->depth + 1);
                    leaf->children.push_back(child);
                    ++move_idx;
                }
                leaf->is_expanded = true;

                // Backpropagate (also undo virtual loss)
                Node* node = leaf;
                while (node != nullptr) {
                    if (node->magic != 0x12345678) {
                        std::cerr << "FATAL: expand_batch backprop encountered corrupted node" << std::endl;
                        break;
                    }
                    node->visit_count++;
                    node->value_sum += val;
                    if (node->virtual_loss > 0) node->virtual_loss--;
                    val = -val;
                    node = node->parent;
                }
            }
        }
    }

    Node* get_root() { return root_; }

    bool advance_root(int from_sq, int to_sq) {
        if (!root_ || !root_->is_expanded) return false;
        chess::Move target = chess::Move::make(
            chess::Square(from_sq), chess::Square(to_sq));
        for (Node* child : root_->children) {
            if (child->move == target) {
                child->parent = nullptr;
                child->depth = 0;  // New root gets depth 0 for Dirichlet noise
                root_ = child;
                return true;
            }
        }
        return false;
    }

    Node* select_leaf_vl() {
        Node* node = root_;
        if (!node) return nullptr;

        while (node->is_expanded && !node->children.empty()) {
            // Hijack check
            if (node->depth > 0 && node->visit_count >= 5 && !node->verified) {
                node->virtual_loss++;
                return node;
            }

            Node* best_child = nullptr;
            float best_score = -1e9;
            int parent_effective_visits = node->visit_count + node->virtual_loss;

            for (Node* child : node->children) {
                if (!child || child->magic != 0x12345678) continue;
                float score = child->ucb_score(c_puct_, parent_effective_visits);
                if (score > best_score) {
                    best_score = score;
                    best_child = child;
                }
            }
            if (!best_child) break;
            node->virtual_loss++;
            node = best_child;
        }
        node->virtual_loss++;
        return node;
    }

    static void select_leaves_batch_vl(
        std::vector<MCTS*>& trees,
        int k,
        std::vector<Node*>& out_leaves,
        std::vector<int>& out_tree_indices
    ) {
        out_leaves.clear();
        out_tree_indices.clear();
        out_leaves.reserve(trees.size() * k);
        out_tree_indices.reserve(trees.size() * k);
        // Note: std::vector is not thread-safe for push_back.
        // We need to pre-allocate and assign by index if using OpenMP.
        // Flattened index mapping: idx = t * k + j

        out_leaves.resize(trees.size() * k);
        out_tree_indices.resize(trees.size() * k);

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (int t = 0; t < (int)trees.size(); t++) {
            for (int j = 0; j < k; j++) {
                Node* leaf = trees[t]->select_leaf_vl();
                int idx = t * k + j;
                out_leaves[idx] = leaf;
                out_tree_indices[idx] = t;
            }
        }
    }

    void undo_virtual_loss_and_update(Node* node, float value) {
        if (!node) return;
        node->verified = true;
        Node* cur = node;
        while (cur != nullptr) {
            cur->visit_count++;
            cur->value_sum += value;
            if (cur->virtual_loss > 0) cur->virtual_loss--;
            value = -value;
            cur = cur->parent;
        }
    }

    void undo_virtual_loss(Node* node) {
        if (!node) return;
        Node* cur = node;
        while (cur != nullptr) {
            if (cur->virtual_loss > 0) cur->virtual_loss--;
            cur = cur->parent;
        }
    }

    void backpropagate(Node* node, float value) {
        while (node != nullptr) {
            if (node->magic != 0x12345678) {
                std::cerr << "FATAL: backpropagate encountered corrupted node" << std::endl;
                break;
            }
            node->visit_count++;
            node->value_sum += value;
            value = -value;
            node = node->parent;
        }
    }

    std::string get_fen(Node* node) {
        if (!node || node->magic != 0x12345678) {
            std::cerr << "FATAL: get_fen called on corrupted node" << std::endl;
            return "";
        }
        return node->board.getFen();
    }

    int game_status(Node* node) {
        if (!node || node->magic != 0x12345678) {
            std::cerr << "FATAL: game_status called on corrupted node" << std::endl;
            return 0;
        }
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, node->board);
        if (moves.empty()) {
            if (node->board.inCheck()) return 1;
            return 2;
        }
        return 0;
    }

    std::vector<std::pair<int, int>> get_root_counts() {
        std::vector<std::pair<int, int>> counts;
        if (!root_) return counts;
        if (root_->magic != 0x12345678) {
            std::cerr << "FATAL: get_root_counts root corrupted" << std::endl;
            return counts;
        }

        for (Node* child : root_->children) {
            if (!child || child->magic != 0x12345678) {
                std::cerr << "FATAL: get_root_counts child corrupted" << std::endl;
                continue;
            }
            int idx = move_to_action_idx(child->move, root_->board.sideToMove());
            counts.push_back({idx, child->visit_count});
        }
        return counts;
    }

    float get_root_value() {
        return root_ ? root_->value() : 0.0f;
    }

    int get_root_visits() { return root_ ? root_->visit_count : 0; }
    size_t get_pool_size() { return pool_.size(); }

    // Expose pool for static expand_batch
    NodePool& pool() { return pool_; }

private:
    Node* root_ = nullptr;  // Raw pointer, pool owns the memory
    float c_puct_;
    int num_simulations_;
    float dirichlet_alpha_;
    float dirichlet_epsilon_;
    NodePool pool_;
    std::mt19937 rng_;

    std::vector<float> sample_dirichlet(int size, float alpha) {
        std::vector<float> samples(size, 0.0f);
        if (size <= 0 || alpha <= 0.0f) return samples;

        std::gamma_distribution<float> gamma(alpha, 1.0f);
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            samples[i] = gamma(rng_);
            sum += samples[i];
        }

        if (sum <= 0.0f) {
            float uniform = 1.0f / static_cast<float>(size);
            std::fill(samples.begin(), samples.end(), uniform);
            return samples;
        }

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; ++i) {
            samples[i] *= inv_sum;
        }
        return samples;
    }
};

// ============================================================================
// Encoding functions (unchanged)
// ============================================================================
inline void fill_plane(float* encoded_data, int batch_idx, int plane_idx, uint64_t bb, int perspective, int max_idx) {
    while (bb) {
        int sq = __builtin_ctzll(bb);
        int r = sq / 8;
        int c = sq % 8;

        if (perspective == 1) {
            r = 7 - r;
            c = 7 - c;
        }

        int idx = batch_idx * (116 * 64) + plane_idx * 64 + r * 8 + c;
        if (idx >= 0 && idx < max_idx) {
            encoded_data[idx] = 1.0f;
        }

        bb &= bb - 1;
    }
}

void encode_single_node(Node* node, float* data, int batch_idx, int max_size) {
    if (!node) return;
    if (node->magic != 0x12345678) {
        std::cerr << "FATAL: Encode called on corrupted node" << std::endl;
        return;
    }

    if (!data) return;

    int perspective = (int)node->board.sideToMove();
    int total_features = 116 * 64;

    // 1. History
    Node* current = node;
    for (int t = 0; t < 8; ++t) {
        if (!current || current->magic != 0x12345678) {
            // If current is null or corrupted, stop traversal
            // (shouldn't happen, but safety check)
            break;
        }

        int base_plane = t * 12;
        const auto& board = current->board;

        for (int pt = 0; pt < 6; ++pt) {
            uint64_t bb = board.pieces(static_cast<chess::PieceType::underlying>(pt), static_cast<chess::Color::underlying>(perspective)).getBits();
            fill_plane(data, batch_idx, base_plane + pt, bb, perspective, max_size);
        }

        for (int pt = 0; pt < 6; ++pt) {
            uint64_t bb = board.pieces(static_cast<chess::PieceType::underlying>(pt), static_cast<chess::Color::underlying>(1 - perspective)).getBits();
            fill_plane(data, batch_idx, base_plane + 6 + pt, bb, perspective, max_size);
        }

        // Duplicate earliest position instead of leaving zeros
        if (current->parent) {
            current = current->parent;
        }
        // else: keep encoding same board for remaining slots
    }

    // 2. Metadata
    int meta_base = 96;
    const auto& b = node->board;

    if (b.enpassantSq() != chess::Square::NO_SQ) {
        int sq = b.enpassantSq().index();
        int r = sq / 8;
        int c = sq % 8;
        if (perspective == 1) { r = 7 - r; c = 7 - c; }
        int idx = batch_idx * total_features + (meta_base + 4) * 64 + r * 8 + c;
        if (idx >= 0 && idx < max_size) data[idx] = 1.0f;
    }

    float halfmove = std::min(b.halfMoveClock() / 50.0f, 1.0f);
    int p_idx = meta_base + 5;
    for (int i = 0; i < 64; ++i) {
        int idx = batch_idx * total_features + p_idx * 64 + i;
        if (idx < max_size) data[idx] = halfmove;
    }

    float fullmove = std::min(b.fullMoveNumber() / 200.0f, 1.0f);
    p_idx = meta_base + 6;
    for (int i = 0; i < 64; ++i) {
        int idx = batch_idx * total_features + p_idx * 64 + i;
        if (idx < max_size) data[idx] = fullmove;
    }

    if (perspective == 1) {
        p_idx = meta_base + 7;
        for (int i = 0; i < 64; ++i) {
            int idx = batch_idx * total_features + p_idx * 64 + i;
            if (idx < max_size) data[idx] = 1.0f;
        }
    }

    // 3. Attack/Defense Maps (planes 104-115)
    int attack_base = 104;
    auto occ = b.occ();
    auto my_color = static_cast<chess::Color::underlying>(perspective);
    auto enemy_color = static_cast<chess::Color::underlying>(1 - perspective);

    // Helper to get attacks for a piece type at a given square
    auto get_attacks = [&](int pt, chess::Square sq, chess::Color color) -> uint64_t {
        using PT = chess::PieceType::underlying;
        switch (static_cast<PT>(pt)) {
            case PT::PAWN:   return chess::attacks::pawn(color, sq).getBits();
            case PT::KNIGHT: return chess::attacks::knight(sq).getBits();
            case PT::BISHOP: return chess::attacks::bishop(sq, occ).getBits();
            case PT::ROOK:   return chess::attacks::rook(sq, occ).getBits();
            case PT::QUEEN:  return chess::attacks::queen(sq, occ).getBits();
            case PT::KING:   return chess::attacks::king(sq).getBits();
            default:         return 0;
        }
    };

    // My attacks (6 planes: attack_base + 0..5)
    for (int pt = 0; pt < 6; ++pt) {
        uint64_t pieces_bb = b.pieces(static_cast<chess::PieceType::underlying>(pt), my_color).getBits();
        uint64_t combined_attacks = 0;
        while (pieces_bb) {
            int sq = __builtin_ctzll(pieces_bb);
            combined_attacks |= get_attacks(pt, chess::Square(sq), chess::Color(my_color));
            pieces_bb &= pieces_bb - 1;
        }
        if (combined_attacks) {
            fill_plane(data, batch_idx, attack_base + pt, combined_attacks, perspective, max_size);
        }
    }

    // Enemy attacks (6 planes: attack_base + 6..11)
    for (int pt = 0; pt < 6; ++pt) {
        uint64_t pieces_bb = b.pieces(static_cast<chess::PieceType::underlying>(pt), enemy_color).getBits();
        uint64_t combined_attacks = 0;
        while (pieces_bb) {
            int sq = __builtin_ctzll(pieces_bb);
            combined_attacks |= get_attacks(pt, chess::Square(sq), chess::Color(enemy_color));
            pieces_bb &= pieces_bb - 1;
        }
        if (combined_attacks) {
            fill_plane(data, batch_idx, attack_base + 6 + pt, combined_attacks, perspective, max_size);
        }
    }
}

} // namespace mcts
