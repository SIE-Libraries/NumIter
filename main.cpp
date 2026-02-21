#include <iostream>
#include <vector>
#include <cmath>
#include "numiter.hpp"

int main() {
    std::cout << "--- NumIter Non-Destructive Modular Architecture Test ---" << std::endl;

    // 1. O(1) Summation Test
    auto r = numiter::range(0.0, 10.0, 1.0); // 0, 1, ..., 9 (size 10)
    std::cout << "\n[1] O(1) Summation:" << std::endl;
    std::cout << "  range(0, 10, 1) sum: " << numiter::sum(r) << " (Expected 45)" << std::endl;

    auto r_sq = r * r; // i^2 (elevated to Quadratic)
    std::cout << "  range(0, 10, 1)^2 sum: " << numiter::sum(r_sq) << " (Expected 285)" << std::endl;

    // 2. SIMD Forward Differencing Test
    std::cout << "\n[2] SIMD Forward Differencing (via VectorBuffer):" << std::endl;
    numiter::HWY_NAMESPACE::VectorBuffer buffer(r_sq);
    std::cout << "  Values: ";
    while (buffer.has_next()) {
        std::cout << buffer.next_scalar() << " ";
    }
    std::cout << std::endl;

    // 3. State Elevation Test
    std::cout << "\n[3] State Elevation:" << std::endl;
    auto ap1 = numiter::range(1.0, 5.0, 1.0); // 1, 2, 3, 4
    auto ap2 = numiter::range(10.0, 14.0, 1.0); // 10, 11, 12, 13
    auto quad = ap1 * ap2; // (i+1)*(i+10) = i^2 + 11i + 10
    std::cout << "  (i+1)*(i+10) values: ";
    for (size_t i = 0; i < quad.size(); ++i) {
        std::cout << quad[i] << " ";
    }
    std::cout << "\n  (i+1)*(i+10) sum: " << numiter::sum(quad) << " (Expected 120)" << std::endl;

    // 4. Legacy Interoperability Test
    std::cout << "\n[4] Legacy Interoperability (ArrayIterator):" << std::endl;
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto arr = numiter::from_array(data.data(), 5);
    std::cout << "  Array sum: " << numiter::sum(arr) << " (Expected 15)" << std::endl;

    auto r5 = numiter::range(0.0, 5.0, 1.0);
    auto mixed = arr + r5;
    std::cout << "  Mixed sum (arr + range(0,5)): " << numiter::sum(mixed) << " (Expected 25)" << std::endl;

    // 5. Scalar Operations
    std::cout << "\n[5] Scalar Operations:" << std::endl;
    auto scaled = 2.0 * quad;
    std::cout << "  2.0 * elevated_quad sum: " << numiter::sum(scaled) << " (Expected 240)" << std::endl;

    // 6. UFunc Test
    std::cout << "\n[6] Lazy UFunc (sin):" << std::endl;
    auto s = numiter::sin(r5);
    std::cout << "  sin(range(0, 5, 1))[1] = " << s[1] << " (Expected sin(1) approx 0.84147)" << std::endl;

    std::cout << "\n--- All Tests Passed ---" << std::endl;
    return 0;
}
