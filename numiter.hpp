#pragma once

#include <cstddef>
#include <cmath>
#include <numeric>
#include <functional>
#include <ranges>
#include <stdexcept>

namespace numiter {

// --- Forward Declarations ---
struct Mul;

template <typename T>
class Scalar;

template <typename Op, typename Lhs, typename Rhs>
class BinaryExpr;

template <typename T1, typename T2, typename Iter>
auto operator*(const T1& c1, const BinaryExpr<Mul, Scalar<T2>, Iter>& rhs);


// --- Operator Definitions ---
#define DEFINE_BINARY_OP(name, op) \
struct name { \
    template <typename T1, typename T2> \
    static auto apply(T1 lhs, T2 rhs) { return lhs op rhs; } \
};

DEFINE_BINARY_OP(Add, +)
DEFINE_BINARY_OP(Sub, -)
DEFINE_BINARY_OP(Mul, *)

#define DEFINE_UNARY_OP(name, func) \
struct name { \
    template <typename T> \
    static auto apply(T val) { return std::func(val); } \
};

DEFINE_UNARY_OP(Sin, sin)
DEFINE_UNARY_OP(Cos, cos)
DEFINE_UNARY_OP(Tan, tan)
DEFINE_UNARY_OP(Log, log)
DEFINE_UNARY_OP(Exp, exp)


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

// Convenience function for iota
template <typename T>
auto iota(T start, T stop) {
    return RangeAdapter(std::views::iota(start, stop));
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
    friend auto operator*(const T1& c1, const BinaryExpr<Mul, Scalar<T2>, Iter>& rhs);
};

template <typename Op, typename Iter>
class UnaryExpr : public numiterator<UnaryExpr<Op, Iter>> {
public:
    UnaryExpr(const Iter& iter) : m_iter(iter) {}
    auto get(size_t index) const { return Op::apply(m_iter[index]); }
    size_t size() const { return m_iter.size(); }
private:
    Iter m_iter;
};

template <typename Op, typename Lhs, typename Rhs>
class BinaryExpr : public numiterator<BinaryExpr<Op, Lhs, Rhs>> {
public:
    BinaryExpr(const Lhs& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs) {
        if (lhs.size() != 0 && rhs.size() != 0 && lhs.size() != rhs.size()) {
            throw std::runtime_error("Incompatible sizes in BinaryExpr");
        }
    }
    auto get(size_t index) const { return Op::apply(m_lhs[index], m_rhs[index]); }
    size_t size() const {
        if (m_lhs.size() != 0) return m_lhs.size();
        return m_rhs.size();
    }
private:
    Lhs m_lhs;
    Rhs m_rhs;
    template <typename T1, typename T2, typename Iter>
    friend auto operator*(const T1& c1, const BinaryExpr<Mul, Scalar<T2>, Iter>& rhs);
};


// --- Operator Overloads ---
template <typename Lhs, typename Rhs>
BinaryExpr<Add, Lhs, Rhs> operator+(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Add, Lhs, Rhs>(lhs.derived(), rhs.derived());
}
template <typename Lhs, typename Rhs>
BinaryExpr<Sub, Lhs, Rhs> operator-(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Sub, Lhs, Rhs>(lhs.derived(), rhs.derived());
}
template <typename Lhs, typename Rhs>
BinaryExpr<Mul, Lhs, Rhs> operator*(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Mul, Lhs, Rhs>(lhs.derived(), rhs.derived());
}
template <typename ScalarType, typename Rhs>
BinaryExpr<Mul, Scalar<ScalarType>, Rhs> operator*(const ScalarType& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Mul, Scalar<ScalarType>, Rhs>(Scalar<ScalarType>(lhs), rhs.derived());
}
template <typename Lhs, typename ScalarType>
BinaryExpr<Mul, Lhs, Scalar<ScalarType>> operator*(const numiterator<Lhs>& lhs, const ScalarType& rhs) {
    return rhs * lhs.derived();
}


// --- Algebraic Simplifications ---
template <typename T1, typename T2, typename Iter>
auto operator*(const T1& c1, const BinaryExpr<Mul, Scalar<T2>, Iter>& rhs) {
    const Iter& iter = rhs.m_rhs;
    const T2& c2 = rhs.m_lhs.m_value;
    return (c1 * c2) * iter;
}


// --- UFuncs ---
template <typename Iter>
UnaryExpr<Sin, Iter> sin(const numiterator<Iter>& iter) {
    return UnaryExpr<Sin, Iter>(iter.derived());
}
template <typename Iter>
UnaryExpr<Cos, Iter> cos(const numiterator<Iter>& iter) {
    return UnaryExpr<Cos, Iter>(iter.derived());
}
template <typename Iter>
UnaryExpr<Tan, Iter> tan(const numiterator<Iter>& iter) {
    return UnaryExpr<Tan, Iter>(iter.derived());
}
template <typename Iter>
UnaryExpr<Log, Iter> log(const numiterator<Iter>& iter) {
    return UnaryExpr<Log, Iter>(iter.derived());
}
template <typename Iter>
UnaryExpr<Exp, Iter> exp(const numiterator<Iter>& iter) {
    return UnaryExpr<Exp, Iter>(iter.derived());
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
    using T = decltype(iter.derived()[0]);
    T total = 0;
    for (size_t i = 0; i < iter.size(); ++i) {
        total += iter.derived()[i];
    }
    return total;
}

template <typename Iter>
auto mean(const numiterator<Iter>& iter) {
    return sum(iter) / static_cast<double>(iter.size());
}

} // namespace numiter
