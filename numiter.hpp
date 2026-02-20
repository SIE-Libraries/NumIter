#pragma once

#include <cstddef>
#include <cmath>
#include <numeric>
#include <functional>
#include <ranges>
#include <stdexcept>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "numiter.hpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

namespace numiter {

// --- Forward Declarations ---
template <typename T>
class Scalar;

template <typename T>
class Quadratic;

template <typename T>
class ArithmeticProgression;

template <typename T>
struct is_quadratic : std::false_type {};

template <typename T>
struct is_quadratic<Quadratic<T>> : std::true_type {};

template <typename T>
struct is_ap : std::false_type {};

template <typename T>
struct is_ap<ArithmeticProgression<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_quadratic_v = is_quadratic<T>::value || is_ap<T>::value;

template <typename T>
inline constexpr bool is_ap_v = is_ap<T>::value;

template <typename Op, typename Lhs, typename Rhs>
class BinaryExpr;

template <typename T1, typename T2, typename Iter>
auto operator*(const T1& c1, const BinaryExpr<std::multiplies<>, Scalar<T2>, Iter>& rhs);


// --- Core numiterator Interface ---
template <typename Derived>
class numiterator {
public:
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }
    auto operator[](size_t index) const {
        return derived().get(index);
    }
    size_t size() const {
        return derived().size();
    }
};


// --- Concrete Iterators and Adapters ---

// Adapter to wrap any C++20 random_access_range
template <std::ranges::random_access_range R>
class RangeAdapter : public numiterator<RangeAdapter<R>> {
public:
    RangeAdapter(R range) : m_range(range) {}
    auto get(size_t index) const { return m_range[index]; }
    size_t size() const { return std::ranges::size(m_range); }
private:
    R m_range;
};

// --- Arithmetic Progression Generator ---
template <typename T>
class ArithmeticProgression : public numiterator<ArithmeticProgression<T>> {
public:
    using value_type = T;
    ArithmeticProgression(T start, T stride, size_t size)
        : m_start(start), m_stride(stride), m_size(size) {}

    T get(size_t i) const { return static_cast<T>(m_start + i * m_stride); }
    size_t size() const { return m_size; }

    T a() const { return 0; }
    T b() const { return m_stride; }
    T c() const { return m_start; }

private:
    T m_start, m_stride;
    size_t m_size;
};

// --- Quadratic Progression Generator ---
template <typename T>
class Quadratic : public numiterator<Quadratic<T>> {
public:
    using value_type = T;
    Quadratic(T a, T b, T c, size_t size)
        : m_a(a), m_b(b), m_c(c), m_size(size) {}

    T get(size_t i) const {
        return static_cast<T>(m_a * i * i + m_b * i + m_c);
    }

    size_t size() const { return m_size; }

    T a() const { return m_a; }
    T b() const { return m_b; }
    T c() const { return m_c; }

private:
    T m_a, m_b, m_c;
    size_t m_size;
};

// Convenience functions
template <typename T>
auto quadratic(T a, T b, T c, size_t size) {
    return Quadratic<T>(a, b, c, size);
}

template <typename T>
auto ap(T start, T stride, size_t size) {
    return ArithmeticProgression<T>(start, stride, size);
}

// Convenience function for iota
template <typename T>
auto iota(T start, T stop) {
    size_t count = (stop > start) ? static_cast<size_t>(stop - start) : 0;
    return ap(start, static_cast<T>(1), count);
}

// Python-like range function with stride
template <typename T>
auto range(T start, T stop, T stride = 1) {
    if (stride == 0) throw std::invalid_argument("stride must not be zero");

    size_t count = 0;
    if ((stride > 0 && stop > start) || (stride < 0 && stop < start)) {
        double d_count = std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(stride) - 1e-10);
        if (d_count > 0) {
            count = static_cast<size_t>(d_count);
        }
    }

    return ap(start, stride, count);
}

template <typename T>
class ArrayIterator : public numiterator<ArrayIterator<T>> {
public:
    ArrayIterator(const T* data, size_t size, size_t stride = 1)
        : m_data(data), m_size(size), m_stride(stride) {}
    T get(size_t index) const { return m_data[index * m_stride]; }
    size_t size() const { return m_size; }
private:
    const T* m_data;
    size_t m_size;
    size_t m_stride;
};

// Convenience function
template <typename T>
ArrayIterator<T> from_array(const T* data, size_t size, size_t stride = 1) {
    return ArrayIterator<T>(data, size, stride);
}


// --- Expression Templates ---

template <typename T>
class Scalar : public numiterator<Scalar<T>> {
public:
    Scalar(T value) : m_value(value) {}
    T get(size_t /*index*/) const { return m_value; }
    size_t size() const { return 0; }
private:
    T m_value;
    template <typename T1, typename T2, typename Iter>
    friend auto operator*(const T1& c1, const BinaryExpr<std::multiplies<>, Scalar<T2>, Iter>& rhs);
};

template <typename Op, typename Iter>
class UnaryExpr : public numiterator<UnaryExpr<Op, Iter>> {
public:
    UnaryExpr(Op op, const Iter& iter) : m_op(op), m_iter(iter) {}
    auto get(size_t index) const { return m_op(m_iter[index]); }
    size_t size() const { return m_iter.size(); }
private:
    Op m_op;
    Iter m_iter;
};

template <typename Op, typename Lhs, typename Rhs>
class BinaryExpr : public numiterator<BinaryExpr<Op, Lhs, Rhs>> {
public:
    BinaryExpr(Op op, const Lhs& lhs, const Rhs& rhs) : m_op(op), m_lhs(lhs), m_rhs(rhs) {
        if (lhs.size() != 0 && rhs.size() != 0 && lhs.size() != rhs.size()) {
            throw std::runtime_error("Incompatible sizes in BinaryExpr");
        }
    }
    auto get(size_t index) const { return m_op(m_lhs[index], m_rhs[index]); }
    size_t size() const {
        if (m_lhs.size() != 0) return m_lhs.size();
        return m_rhs.size();
    }
private:
    Op m_op;
    Lhs m_lhs;
    Rhs m_rhs;
    template <typename T1, typename T2, typename Iter>
    friend auto operator*(const T1& c1, const BinaryExpr<std::multiplies<>, Scalar<T2>, Iter>& rhs);
};


// --- Operator Overloads ---
template <typename Lhs, typename Rhs>
auto operator+(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    if constexpr (is_quadratic_v<Lhs> && is_quadratic_v<Rhs>) {
        auto& q1 = lhs.derived();
        auto& q2 = rhs.derived();
        if (q1.size() != 0 && q2.size() != 0 && q1.size() != q2.size()) {
             throw std::runtime_error("Incompatible sizes in BinaryExpr (Quadratic + Quadratic)");
        }
        size_t new_size = q1.size() != 0 ? q1.size() : q2.size();
        return quadratic(q1.a() + q2.a(), q1.b() + q2.b(), q1.c() + q2.c(), new_size);
    } else {
        return BinaryExpr(std::plus<>{}, lhs.derived(), rhs.derived());
    }
}
template <typename Lhs, typename Rhs>
auto operator-(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    if constexpr (is_quadratic_v<Lhs> && is_quadratic_v<Rhs>) {
        auto& q1 = lhs.derived();
        auto& q2 = rhs.derived();
        if (q1.size() != 0 && q2.size() != 0 && q1.size() != q2.size()) {
             throw std::runtime_error("Incompatible sizes in BinaryExpr (Quadratic - Quadratic)");
        }
        size_t new_size = q1.size() != 0 ? q1.size() : q2.size();
        return quadratic(q1.a() - q2.a(), q1.b() - q2.b(), q1.c() - q2.c(), new_size);
    } else {
        return BinaryExpr(std::minus<>{}, lhs.derived(), rhs.derived());
    }
}
template <typename Lhs, typename Rhs>
auto operator*(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    if constexpr (is_ap_v<Lhs> && is_ap_v<Rhs>) {
        auto& q1 = lhs.derived();
        auto& q2 = rhs.derived();
        if (q1.size() != 0 && q2.size() != 0 && q1.size() != q2.size()) {
             throw std::runtime_error("Incompatible sizes in BinaryExpr (AP * AP)");
        }
        size_t new_size = q1.size() != 0 ? q1.size() : q2.size();
        return quadratic(q1.b() * q2.b(),
                        q1.b() * q2.c() + q2.b() * q1.c(),
                        q1.c() * q2.c(),
                        new_size);
    } else {
        return BinaryExpr(std::multiplies<>{}, lhs.derived(), rhs.derived());
    }
}
template <typename T>
concept NotNumIterator = !std::is_base_of_v<numiterator<T>, T>;

template <NotNumIterator ScalarType, typename Rhs>
auto operator*(const ScalarType& lhs, const numiterator<Rhs>& rhs) {
    if constexpr (is_quadratic_v<Rhs>) {
        auto& q = rhs.derived();
        return quadratic(static_cast<typename Rhs::value_type>(lhs * q.a()),
                        static_cast<typename Rhs::value_type>(lhs * q.b()),
                        static_cast<typename Rhs::value_type>(lhs * q.c()),
                        q.size());
    } else {
        return BinaryExpr(std::multiplies<>{}, Scalar<ScalarType>(lhs), rhs.derived());
    }
}
template <typename Lhs, NotNumIterator ScalarType>
auto operator*(const numiterator<Lhs>& lhs, const ScalarType& rhs) {
    return rhs * lhs.derived();
}


// --- Algebraic Simplifications ---
template <typename T1, typename T2, typename Iter>
auto operator*(const T1& c1, const BinaryExpr<std::multiplies<>, Scalar<T2>, Iter>& rhs) {
    const Iter& iter = rhs.m_rhs;
    const T2& c2 = rhs.m_lhs.m_value;
    return (c1 * c2) * iter;
}


// --- UFuncs ---
template <typename Iter>
auto sin(const numiterator<Iter>& iter) {
    return UnaryExpr([](auto x) { return std::sin(x); }, iter.derived());
}
template <typename Iter>
auto cos(const numiterator<Iter>& iter) {
    return UnaryExpr([](auto x) { return std::cos(x); }, iter.derived());
}
template <typename Iter>
auto tan(const numiterator<Iter>& iter) {
    return UnaryExpr([](auto x) { return std::tan(x); }, iter.derived());
}
template <typename Iter>
auto log(const numiterator<Iter>& iter) {
    return UnaryExpr([](auto x) { return std::log(x); }, iter.derived());
}
template <typename Iter>
auto exp(const numiterator<Iter>& iter) {
    return UnaryExpr([](auto x) { return std::exp(x); }, iter.derived());
}
template <typename Iter>
auto sqrt(const numiterator<Iter>& iter) {
    return UnaryExpr([](auto x) { return std::sqrt(x); }, iter.derived());
}
template <typename Iter>
auto abs(const numiterator<Iter>& iter) {
    return UnaryExpr([](auto x) { return std::abs(x); }, iter.derived());
}

template <typename Lhs, typename Rhs>
auto pow(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr([](auto x, auto y) { return std::pow(x, y); }, lhs.derived(), rhs.derived());
}

template <typename Lhs, typename T>
auto pow(const numiterator<Lhs>& lhs, T exponent) {
    return BinaryExpr([](auto x, auto y) { return std::pow(x, y); }, lhs.derived(), Scalar<T>(exponent));
}


// --- 3D View ---
template <typename Iter>
class View3D {
public:
    View3D(const Iter& iter, size_t dim_i, size_t dim_j, size_t dim_k)
        : m_iter(iter), m_dim_i(dim_i), m_dim_j(dim_j), m_dim_k(dim_k) {
        if (iter.size() != dim_i * dim_j * dim_k) {
            // In a real library, we'd throw an exception here.
        }
    }
    auto get(size_t i, size_t j, size_t k) const {
        return m_iter[i * m_dim_j * m_dim_k + j * m_dim_k + k];
    }
    size_t size() const { return m_iter.size(); }
private:
    Iter m_iter;
    size_t m_dim_i;
    size_t m_dim_j;
    size_t m_dim_k;
};


// --- Reductions ---
template <typename Iter>
auto sum(const numiterator<Iter>& iter) {
    if constexpr (is_quadratic_v<Iter>) {
        auto& q = iter.derived();
        using T = typename Iter::value_type;
        double n = static_cast<double>(q.size());
        if (n == 0) return static_cast<T>(0);
        double A = static_cast<double>(q.a());
        double B = static_cast<double>(q.b());
        double C = static_cast<double>(q.c());

        // Sum = A * (n-1)n(2n-1)/6 + B * (n-1)n/2 + C * n
        double result = A * (n - 1.0) * n * (2.0 * n - 1.0) / 6.0 +
                        B * (n - 1.0) * n / 2.0 +
                        C * n;
        return static_cast<T>(result);
    } else {
        using T = decltype(iter.derived()[0]);
        T total = 0;
        for (size_t i = 0; i < iter.size(); ++i) {
            total += iter.derived()[i];
        }
        return total;
    }
}

template <typename Iter>
auto mean(const numiterator<Iter>& iter) {
    return sum(iter) / static_cast<double>(iter.size());
}

template <typename Iter>
auto product(const numiterator<Iter>& iter) {
    using T = decltype(iter.derived()[0]);
    T prod = 1;
    for (size_t i = 0; i < iter.size(); ++i) {
        prod *= iter.derived()[i];
    }
    return prod;
}

template <typename Iter>
auto min(const numiterator<Iter>& iter) {
    if (iter.size() == 0) throw std::runtime_error("min of empty iterator");
    auto m = iter.derived()[0];
    for (size_t i = 1; i < iter.size(); ++i) {
        auto val = iter.derived()[i];
        if (val < m) m = val;
    }
    return m;
}

template <typename Iter>
auto max(const numiterator<Iter>& iter) {
    if (iter.size() == 0) throw std::runtime_error("max of empty iterator");
    auto m = iter.derived()[0];
    for (size_t i = 1; i < iter.size(); ++i) {
        auto val = iter.derived()[i];
        if (val > m) m = val;
    }
    return m;
}


// --- Generic Mapping and Zipping ---

/**
 * @brief Applies a unary function to each element of the iterator lazily.
 */
template <typename Iter, typename F>
auto map(const numiterator<Iter>& iter, F f) {
    return UnaryExpr(f, iter.derived());
}

/**
 * @brief Applies a binary function to elements of two iterators lazily.
 */
template <typename Lhs, typename Rhs, typename F>
auto zip(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs, F f) {
    return BinaryExpr(f, lhs.derived(), rhs.derived());
}

} // namespace numiter
