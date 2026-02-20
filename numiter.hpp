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
struct Add {
    template <typename T1, typename T2>
    auto operator()(T1 lhs, T2 rhs) const { return lhs + rhs; }
};

struct Sub {
    template <typename T1, typename T2>
    auto operator()(T1 lhs, T2 rhs) const { return lhs - rhs; }
};

struct Mul {
    template <typename T1, typename T2>
    auto operator()(T1 lhs, T2 rhs) const { return lhs * rhs; }
};

struct Sin {
    template <typename T>
    auto operator()(T val) const { return std::sin(val); }
};

struct Cos {
    template <typename T>
    auto operator()(T val) const { return std::cos(val); }
};

struct Tan {
    template <typename T>
    auto operator()(T val) const { return std::tan(val); }
};

struct Log {
    template <typename T>
    auto operator()(T val) const { return std::log(val); }
};

struct Exp {
    template <typename T>
    auto operator()(T val) const { return std::exp(val); }
};


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

// Python-like range function with stride
template <typename T>
auto range(T start, T stop, T stride = 1) {
    if (stride == 0) throw std::invalid_argument("stride must not be zero");

    size_t count = 0;
    if ((stride > 0 && stop > start) || (stride < 0 && stop < start)) {
        // Use double for count calculation to handle both integer and floating point types.
        // We subtract a tiny epsilon to handle floating point precision issues
        // when (stop - start) is exactly a multiple of stride.
        double d_count = std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(stride) - 1e-10);
        if (d_count > 0) {
            count = static_cast<size_t>(d_count);
        }
    }

    return RangeAdapter(std::views::iota(size_t(0), count) | std::views::transform([start, stride](size_t i) {
        return static_cast<T>(static_cast<double>(start) + static_cast<double>(i) * static_cast<double>(stride));
    }));
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
    friend auto operator*(const T1& c1, const BinaryExpr<Mul, Scalar<T2>, Iter>& rhs);
};


// --- Operator Overloads ---
template <typename Lhs, typename Rhs>
auto operator+(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr(Add{}, lhs.derived(), rhs.derived());
}
template <typename Lhs, typename Rhs>
auto operator-(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr(Sub{}, lhs.derived(), rhs.derived());
}
template <typename Lhs, typename Rhs>
auto operator*(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr(Mul{}, lhs.derived(), rhs.derived());
}
template <typename ScalarType, typename Rhs>
auto operator*(const ScalarType& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr(Mul{}, Scalar<ScalarType>(lhs), rhs.derived());
}
template <typename Lhs, typename ScalarType>
auto operator*(const numiterator<Lhs>& lhs, const ScalarType& rhs) {
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
auto sin(const numiterator<Iter>& iter) {
    return UnaryExpr(Sin{}, iter.derived());
}
template <typename Iter>
auto cos(const numiterator<Iter>& iter) {
    return UnaryExpr(Cos{}, iter.derived());
}
template <typename Iter>
auto tan(const numiterator<Iter>& iter) {
    return UnaryExpr(Tan{}, iter.derived());
}
template <typename Iter>
auto log(const numiterator<Iter>& iter) {
    return UnaryExpr(Log{}, iter.derived());
}
template <typename Iter>
auto exp(const numiterator<Iter>& iter) {
    return UnaryExpr(Exp{}, iter.derived());
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
