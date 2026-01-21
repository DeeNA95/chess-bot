#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mcts.cpp"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
    py::class_<mcts::Node, std::unique_ptr<mcts::Node, py::nodelete>>(m, "Node")
        .def_property_readonly("visit_count", [](mcts::Node& n) { return n.visit_count; })
        .def_property_readonly("value", [](mcts::Node& n) { return n.value(); });

    py::class_<mcts::MCTS>(m, "MCTS")
        .def(py::init<float, int>(), py::arg("c_puct"), py::arg("num_simulations"))
        .def("reset", &mcts::MCTS::reset)
        .def("select_leaf", &mcts::MCTS::select_leaf, py::return_value_policy::reference)
        .def("expand", &mcts::MCTS::expand)
        .def("get_fen", &mcts::MCTS::get_fen)
        .def("game_status", &mcts::MCTS::game_status)
        .def("get_root_counts", &mcts::MCTS::get_root_counts)
        .def("get_root_value", &mcts::MCTS::get_root_value)
        .def("get_root_visits", &mcts::MCTS::get_root_visits);

    m.def("encode_batch", [](std::vector<mcts::Node*> nodes) {
        int batch_size = nodes.size();
        std::vector<ssize_t> shape = {batch_size, 116, 8, 8};
        py::array_t<float> result(shape);

        float* raw_ptr = static_cast<float*>(result.mutable_data());
        std::memset(raw_ptr, 0, batch_size * 116 * 64 * sizeof(float));

        if (batch_size > 0 && batch_size > 4) {
             #pragma omp parallel for
             for (int i = 0; i < batch_size; ++i) {
                 mcts::encode_single_node(nodes[i], raw_ptr, i);
             }
        } else {
             for (int i = 0; i < batch_size; ++i) {
                 mcts::encode_single_node(nodes[i], raw_ptr, i);
             }
        }

        return result;
    });

    m.def("expand_batch_fast", [](
        std::vector<mcts::Node*> leaves,
        py::array_t<float> policy_probs,
        py::array_t<float> values
    ) {
        // Buffer info
        py::buffer_info policy_buf = policy_probs.request();
        py::buffer_info values_buf = values.request();

        float* policy_ptr = static_cast<float*>(policy_buf.ptr);
        float* values_ptr = static_cast<float*>(values_buf.ptr);

        int batch_size = leaves.size();

        // Call static implementation
        mcts::MCTS::expand_batch(leaves, policy_ptr, values_ptr, batch_size);
    });

    m.def("select_leaf_batch", [](std::vector<mcts::MCTS*> trees) {
        std::vector<mcts::Node*> leaves;
        mcts::MCTS::select_leaf_batch(trees, leaves);
        return leaves;
    });
}
