#ifndef NUMITER_HPP
#define NUMITER_HPP

#include <cstddef>
#include <cmath>
#include <numeric>
#include <functional>

namespace numiter {

// Existing code remains untouched...

// New: Add hyperbolic functions
#define DEFINE_HYPERBOLIC_OP(name, func) \
struct name { \
    template <typename T> \
    static auto apply(T val) { return std::func(val); } \
};

DEFINE_HYPERBOLIC_OP(Sinh, sinh);
DEFINE_HYPERBOLIC_OP(Cosh, cosh);
DEFINE_HYPERBOLIC_OP(Tanh, tanh);

// Hyperbolic function operator overloads
template <typename Iter>
UnaryExpr<Sinh, Iter> sinh(const numiterator<Iter>& iter) {
    return UnaryExpr<Sinh, Iter>(iter.derived());
}

template <typename Iter>
UnaryExpr<Cosh, Iter> cosh(const numiterator<Iter>& iter) {
    return UnaryExpr<Cosh, Iter>(iter.derived());
}

template <typename Iter>
UnaryExpr<Tanh, Iter> tanh(const numiterator<Iter>& iter) {
    return UnaryExpr<Tanh, Iter>(iter.derived());
}

// New: Add variance computation template
template <typename Iter>
auto variance(const numiterator<Iter>& iter) {
    auto mean_val = mean(iter);
    double var = 0;
    for (size_t i = 0; i < iter.size(); ++i) {
        var += std::pow(iter.derived()[i] - mean_val, 2);
    }
    return var / static_cast<double>(iter.size());
}

// New: Quadratic and higher-degree iterator
// This supports iterators of the form a*x^2 + b*x + c.
template <typename T>
class polynomial_iterator : public numiterator<polynomial_iterator<T>> {
public:
    polynomial_iterator(T a, T b, T c, T start, T stop, T step = 1)
        : m_a(a), m_b(b), m_c(c), m_start(start), m_stop(stop), m_step(step) {}

    T get(size_t index) const {
        T x = m_start + index * m_step;
        return m_a * x * x + m_b * x + m_c;
    }

    size_t size() const {
        return numiter::range(m_start, m_stop, m_step).size();
    }

private:
    T m_a, m_b, m_c;
    T m_start, m_stop, m_step;
};

} // namespace numiter
#endif // NUMITER_HPP