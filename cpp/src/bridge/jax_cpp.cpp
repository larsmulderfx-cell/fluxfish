#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "jaxlib/kernel_pybind11_helpers.h"

// Zero-copy interface for LRT evaluation
py::array_t<float> evaluate_position_lrt(
    const BitBoard& board,
    py::array_t<float> reasoning_token,
    int max_iterations
) {
    py::gil_scoped_release release; // Release GIL for parallelism
    
    // Direct memory mapping without copies
    auto buf = reasoning_token.request();
    float* token_ptr = static_cast<float*>(buf.ptr);
    
    // Call JAX-compiled function
    return jax::Pjit(lrt_evaluate)(board, token_ptr, max_iterations);
}