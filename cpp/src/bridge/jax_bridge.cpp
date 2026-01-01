#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "liquid_chess/bitboard.hpp"

namespace py = pybind11;

class JAXLRTBridge {
private:
    py::object lrt_model;
    py::object jax_device;
    
    // Cache for board tensors (avoid Python allocation)
    std::unordered_map<uint64_t, py::array_t<float>> tensor_cache;
    
public:
    JAXLRTBridge(const std::string& model_path) {
        py::gil_scoped_acquire acquire;
        
        // Import Python module
        py::module liquid_chess = py::module::import("liquid_chess.models.lrt");
        
        // Load model
        lrt_model = liquid_chess.attr("load_model")(model_path);
        
        // Get JAX device for async execution
        py::module jax = py::module::import("jax");
        jax_device = jax.attr("devices")("cpu")[0];
    }
    
    float evaluate_position(const BitBoard& board) {
        uint64_t key = board.get_key();
        
        // Check cache first
        auto it = tensor_cache.find(key);
        py::array_t<float> board_tensor;
        
        if (it != tensor_cache.end()) {
            board_tensor = it->second;
        } else {
            // Convert board to tensor (zero-copy possible)
            board_tensor = board_to_tensor(board);
            tensor_cache[key] = board_tensor;
            
            // Limit cache size
            if (tensor_cache.size() > 10000) {
                tensor_cache.erase(tensor_cache.begin());
            }
        }
        
        // Async evaluation on JAX device
        py::gil_scoped_acquire acquire;
        
        // Use JIT-compiled fast evaluation
        py::object result = lrt_model.attr("evaluate_fast")(
            board_tensor,
            py::str(std::to_string(key))
        );
        
        return result.cast<float>();
    }
    
private:
    py::array_t<float> board_to_tensor(const BitBoard& board) {
        // Create numpy array without copy
        auto arr = py::array_t<float>({64, 32});  // 64 squares, 32 features
        
        auto buf = arr.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        // Fill tensor directly (fast C++ loop)
        for (int sq = 0; sq < 64; ++sq) {
            // Piece type (one-hot encoded)
            Piece pc = board.piece_on(Square(sq));
            if (pc != NO_PIECE) {
                ptr[sq * 32 + pc] = 1.0f;
            }
            
            // Square features
            ptr[sq * 32 + 12] = rank_of(Square(sq)) / 7.0f;
            ptr[sq * 32 + 13] = file_of(Square(sq)) / 7.0f;
            
            // Attack maps
            ptr[sq * 32 + 14] = board.is_attacked(Square(sq), WHITE) ? 1.0f : 0.0f;
            ptr[sq * 32 + 15] = board.is_attacked(Square(sq), BLACK) ? 1.0f : 0.0f;
            
            // Add more features...
        }
        
        return arr;
    }
};

// PyBind11 module
PYBIND11_MODULE(liquid_chess_cpp, m) {
    m.doc() = "Liquid Chess Engine C++ Core";
    
    py::class_<JAXLRTBridge>(m, "JAXLRTBridge")
        .def(py::init<const std::string&>())
        .def("evaluate_position", &JAXLRTBridge::evaluate_position);
    
    // Expose other C++ classes...
}