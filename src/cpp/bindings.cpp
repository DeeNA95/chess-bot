#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mcts.cpp"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
    py::class_<mcts::Node, std::unique_ptr<mcts::Node, py::nodelete>>(m, "Node")
        .def_property_readonly("visit_count", [](mcts::Node& n) { return n.visit_count; })
        .def_property_readonly("value", [](mcts::Node& n) { return n.value(); })
        .def_property_readonly("depth", [](mcts::Node& n) { return n.depth; })
        .def_property_readonly("is_expanded", [](mcts::Node& n) { return n.is_expanded; })
        .def_property_readonly("verified", [](mcts::Node& n) { return n.verified; });

    py::class_<mcts::MCTS>(m, "MCTS")
        .def(
            py::init<float, int, float, float>(),
            py::arg("c_puct"),
            py::arg("num_simulations"),
            py::arg("dirichlet_alpha") = 0.03f,
            py::arg("dirichlet_epsilon") = 0.0f
        )
        .def("reset", [](mcts::MCTS& m, const std::string& fen) {
            try { m.reset(fen); } catch (const std::exception& e) { std::cerr << "C++ Exception in reset: " << e.what() << std::endl; throw; }
        })
        .def("select_leaf", [](mcts::MCTS& m) {
            try { return m.select_leaf(); } catch (const std::exception& e) { std::cerr << "C++ Exception in select_leaf: " << e.what() << std::endl; throw; }
        }, py::return_value_policy::reference_internal)
        .def("expand", &mcts::MCTS::expand)
        .def("update_value", &mcts::MCTS::update_value)
        .def("get_fen", [](mcts::MCTS& m, mcts::Node* n) {
             try { return m.get_fen(n); } catch (const std::exception& e) { std::cerr << "C++ Exception in get_fen: " << e.what() << std::endl; throw; }
        })
        .def("game_status", &mcts::MCTS::game_status)
        .def("get_root_counts", &mcts::MCTS::get_root_counts)
        .def("get_root_value", &mcts::MCTS::get_root_value)
        .def("get_root_visits", &mcts::MCTS::get_root_visits);

    m.def("encode_batch", [](std::vector<mcts::Node*> nodes) {
        try {
            int batch_size = nodes.size();
            std::vector<ssize_t> shape = {batch_size, 116, 8, 8};
            py::array_t<float> result(shape);

            float* raw_ptr = static_cast<float*>(result.mutable_data());
            int total_size = batch_size * 116 * 64;
            std::memset(raw_ptr, 0, total_size * sizeof(float));

            // Release GIL for the encoding loop
            {
                py::gil_scoped_release release;
                // Sequential loop (OpenMP removed for stability with Python multiprocessing)
                for (int i = 0; i < batch_size; ++i) {
                    if (nodes[i]) mcts::encode_single_node(nodes[i], raw_ptr, i, total_size);
                }
            }
            return result;
        } catch (const std::exception& e) {
            std::cerr << "C++ Exception in encode_batch: " << e.what() << std::endl;
            throw;
        }
    });

    m.def("expand_batch_fast", [](
        std::vector<mcts::MCTS*> trees,
        std::vector<mcts::Node*> leaves,
        py::array_t<float> policy_probs,
        py::array_t<float> values
    ) {
        try {
            // Buffer info
            py::buffer_info policy_buf = policy_probs.request();
            py::buffer_info values_buf = values.request();

            float* policy_ptr = static_cast<float*>(policy_buf.ptr);
            float* values_ptr = static_cast<float*>(values_buf.ptr);

            int batch_size = leaves.size();

            // Validate input sizes
            if (policy_buf.size < batch_size * 4096) {
                 std::cerr << "FATAL: Policy buffer too small! Expected " << batch_size * 4096 << ", got " << policy_buf.size << std::endl;
                 return;
            }
            if (values_buf.size < batch_size) {
                 std::cerr << "FATAL: Values buffer too small! Expected " << batch_size << ", got " << values_buf.size << std::endl;
                 return;
            }

            // Call static implementation with GIL released
            {
                py::gil_scoped_release release;
                mcts::MCTS::expand_batch(trees, leaves, policy_ptr, values_ptr, batch_size);
            }
        } catch (const std::exception& e) {
            std::cerr << "C++ Exception in expand_batch_fast: " << e.what() << std::endl;
            throw;
        }
    });

    m.def("select_leaf_batch", [](std::vector<mcts::MCTS*> trees) {
        try {
            std::vector<mcts::Node*> leaves;
            {
                py::gil_scoped_release release;
                mcts::MCTS::select_leaf_batch(trees, leaves);
            }
            py::list out;
            for (auto* leaf : leaves) {
                out.append(py::cast(leaf, py::return_value_policy::reference));
            }
            return out;
        } catch (const std::exception& e) {
            std::cerr << "C++ Exception in select_leaf_batch: " << e.what() << std::endl;
            throw;
        }
    });
}
