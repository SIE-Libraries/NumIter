#include <iostream>
#include <vector>
#include <cmath> // For std::sin in the final check
#include "numiter.hpp"

/**
 * @brief main() for demonstrating the NumIter library.
 *
 * This file showcases the core features of the NumIter library, including:
 * 1. Lazy evaluation with expression templates.
 * 2. Interoperability with raw C++ arrays (simulating NumPy buffers).
 * 3. Strided access for efficient data manipulation.
 * 4. O(1) algebraic simplifications.
 * 5. 3D coordinate mapping.
 * 6. Eager reductions for computing results.
 * 7. A final example combining these features into a single, optimized expression.
 */
int main() {
    std::cout << "--- NumIter Library Demonstration ---" << std::endl;

    // --- 1. Basic iota iterator ---
    std::cout << "\n[1] Basic iota(0, 5):" << std::endl;
    auto r = numiter::iota(0, 5);
    for (size_t i = 0; i < r.size(); ++i) {
        std::cout << "  r[" << i << "] = " << r[i] << std::endl;
    }

    // --- 2. NumPy-like ArrayIterator with striding ---
    // Simulates a raw NumPy array buffer.
    std::cout << "\n[2] Wrapping a raw C++ array with stride=2:" << std::endl;
    std::vector<double> numpy_sim_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    // Create an iterator that only accesses every 2nd element.
    auto array_iter = numiter::from_array(numpy_sim_data.data(), 3, 2);
    for (size_t i = 0; i < array_iter.size(); ++i) {
        std::cout << "  array_iter[" << i << "] = " << array_iter[i] << std::endl;
    }

    // --- 3. Lazy Expression Templates ---
    // The expression `2.0 * sin(r)` is not computed immediately.
    // It creates an expression object that is evaluated on-demand.
    std::cout << "\n[3] Lazy Expression: 2.0 * sin(iota(0, 5))" << std::endl;
    auto lazy_expr = 2.0 * numiter::sin(r);
    for (size_t i = 0; i < lazy_expr.size(); ++i) {
        std::cout << "  lazy_expr[" << i << "] = " << lazy_expr[i] << std::endl;
    }

    // --- 4. O(1) Algebraic Simplification ---
    // The expression `2.0 * (3.0 * r)` is simplified at compile time
    // to `6.0 * r`.
    std::cout << "\n[4] O(1) Algebraic Simplification: 2.0 * (3.0 * iota(0, 5))" << std::endl;
    auto simplified_expr = 2.0 * (3.0 * r);
    for (size_t i = 0; i < simplified_expr.size(); ++i) {
        std::cout << "  simplified_expr[" << i << "] = " << simplified_expr[i] << std::endl;
    }

    // --- 5. 3D Coordinate Mapping ---
    // Treat a flat iterator as a 3D view.
    std::cout << "\n[5] 3D View (2x3x4) over iota(0, 24):" << std::endl;
    auto r_3d = numiter::iota(0, 2 * 3 * 4);
    numiter::View3D view(r_3d, 2, 3, 4);
    std::cout << "  view.get(1, 1, 1) = " << view.get(1, 1, 1) << std::endl;
    std::cout << "  Expected flat index: " << (1 * 3 * 4 + 1 * 4 + 1) << std::endl;

    // --- 6. Test Subtraction ---
    std::cout << "\n[6] Subtraction: iota(10, 15) - iota(0, 5)" << std::endl;
    auto r1 = numiter::iota(10, 15);
    auto r2 = numiter::iota(0, 5);
    auto sub_expr = r1 - r2;
    for (size_t i = 0; i < sub_expr.size(); ++i) {
        std::cout << "  sub_expr[" << i << "] = " << sub_expr[i] << std::endl;
    }

    // --- 7. Final Example: Optimized NumPy-like Computation ---
    // This demonstrates the library's core strength: a complex mathematical
    // expression on a raw data buffer is compiled into a single, efficient loop.
    // No intermediate arrays are created.
    std::cout << "\n[7] Final Example: mean(sin(2.0 * array_iter))" << std::endl;

    // Use the strided iterator from step 2
    auto final_expr = numiter::sin(2.0 * array_iter);
    double final_mean = numiter::mean(final_expr);

    std::cout << "  Input data (strided): 1.0, 3.0, 5.0" << std::endl;
    std::cout << "  Expression: sin(2.0 * data)" << std::endl;
    std::cout << "  Resulting mean: " << final_mean << std::endl;

    // Manual verification
    double manual_sum = std::sin(2.0 * 1.0) + std::sin(2.0 * 3.0) + std::sin(2.0 * 5.0);
    double manual_mean = manual_sum / 3.0;
    std::cout << "  Manual verification: " << manual_mean << std::endl;

    // --- 8. Test Size Mismatch Exception ---
    std::cout << "\n[8] Testing exception for mismatched sizes:" << std::endl;
    try {
        auto r_long = numiter::iota(0, 10);
        auto r_short = numiter::iota(0, 5);
        auto bad_expr = r_long + r_short;
        // This line should not be reached
        std::cout << "  Error: Exception was not thrown!" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "  Successfully caught expected exception: " << e.what() << std::endl;
    }

    std::cout << "\n--- Demonstration Complete ---" << std::endl;
    return 0;
}
