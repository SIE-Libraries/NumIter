#ifndef NUMITER_HPP
#define NUMITER_HPP

#include <cstddef>
#include <cmath>
#include <numeric>
#include <functional>

namespace numiter {

template <typename Derived>
class numiterator {
public:
    // CRTP access to the derived class
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }

    // Read-only access to elements
    auto operator[](size_t index) const {
        return derived().get(index);
    }

    // Number of elements in the iterator
    size_t size() const {
        return derived().size();
    }
};

} // namespace numiter

namespace numiter {

template <typename T>
class range_iterator : public numiterator<range_iterator<T>> {
public:
    range_iterator(T start, T stop, T step = 1)
        : m_start(start), m_stop(stop), m_step(step) {}

    T get(size_t index) const {
        return m_start + index * m_step;
    }

    size_t size() const {
        if (m_step > 0 && m_stop <= m_start) return 0;
        if (m_step < 0 && m_stop >= m_start) return 0;
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<size_t>(std::ceil((m_stop - m_start) / m_step));
        } else {
            return (m_stop - m_start + m_step - (m_step > 0 ? 1 : -1)) / m_step;
        }
    }

    T m_start;
    T m_stop;
    T m_step;
};

// Convenience function
template <typename T>
range_iterator<T> range(T start, T stop, T step = 1) {
    return range_iterator<T>(start, stop, step);
}

} // namespace numiter

namespace numiter {

template <typename T>
class ArrayIterator : public numiterator<ArrayIterator<T>> {
public:
    ArrayIterator(const T* data, size_t size, size_t stride = 1)
        : m_data(data), m_size(size), m_stride(stride) {}

    T get(size_t index) const {
        return m_data[index * m_stride];
    }

    size_t size() const {
        return m_size;
    }

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

// Scalar wrapper to treat literals as numiterators
template <typename T>
class Scalar : public numiterator<Scalar<T>> {
public:
    Scalar(T value) : m_value(value) {}
    T get(size_t /*index*/) const { return m_value; }
    // Size is conceptually infinite, but for expressions, we take the size of the other operand.
    // A standalone scalar has no size.
    size_t size() const { return 0; }

    T m_value;
};

// Expression template for unary operations (e.g., sin(iter))
template <typename Op, typename Iter>
class UnaryExpr : public numiterator<UnaryExpr<Op, Iter>> {
public:
    UnaryExpr(const Iter& iter) : m_iter(iter) {}
    auto get(size_t index) const { return Op::apply(m_iter[index]); }
    size_t size() const { return m_iter.size(); }
public:
    Iter m_iter;
};

// Expression template for binary operations (e.g., iter1 + iter2)
template <typename Op, typename Lhs, typename Rhs>
class BinaryExpr : public numiterator<BinaryExpr<Op, Lhs, Rhs>> {
public:
    BinaryExpr(const Lhs& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs) {}
    auto get(size_t index) const { return Op::apply(m_lhs[index], m_rhs[index]); }
    size_t size() const {
        if (m_lhs.size() != 0) return m_lhs.size();
        return m_rhs.size();
    }
public:
    Lhs m_lhs;
    Rhs m_rhs;
};

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


// --- Operator Overloads ---
// iter + iter
template <typename Lhs, typename Rhs>
BinaryExpr<Add, Lhs, Rhs> operator+(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Add, Lhs, Rhs>(lhs.derived(), rhs.derived());
}
// iter - iter
template <typename Lhs, typename Rhs>
BinaryExpr<Sub, Lhs, Rhs> operator-(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Sub, Lhs, Rhs>(lhs.derived(), rhs.derived());
}
// iter * iter
template <typename Lhs, typename Rhs>
BinaryExpr<Mul, Lhs, Rhs> operator*(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Mul, Lhs, Rhs>(lhs.derived(), rhs.derived());
}

// scalar * iter
template <typename ScalarType, typename Rhs>
BinaryExpr<Mul, Scalar<ScalarType>, Rhs> operator*(const ScalarType& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr<Mul, Scalar<ScalarType>, Rhs>(Scalar<ScalarType>(lhs), rhs.derived());
}

// iter * scalar
template <typename Lhs, typename ScalarType>
BinaryExpr<Mul, Lhs, Scalar<ScalarType>> operator*(const numiterator<Lhs>& lhs, const ScalarType& rhs) {
    return rhs * lhs.derived();
}

// --- Algebraic Simplifications ---

// c1 * (c2 * iter) -> (c1 * c2) * iter
template <typename T1, typename T2, typename Iter>
auto operator*(const T1& c1, const BinaryExpr<Mul, Scalar<T2>, Iter>& rhs) {
    // Extract the inner iterator and the scalar c2 from the rhs expression
    const Iter& iter = rhs.m_rhs; // This needs friend access or public members
    const T2& c2 = rhs.m_lhs.get(0);
    return (c1 * c2) * iter;
}

// c * range(start, stop, step) -> range(c*start, c*stop, c*step)
template <typename T>
range_iterator<T> operator*(T c, const range_iterator<T>& r) {
    return range(c * r.m_start, c * r.m_stop, c * r.m_step);
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

// 3D View
template <typename Iter>
class View3D {
public:
    View3D(const Iter& iter, size_t dim_i, size_t dim_j, size_t dim_k)
        : m_iter(iter), m_dim_i(dim_i), m_dim_j(dim_j), m_dim_k(dim_k) {
        if (iter.size() != dim_i * dim_j * dim_k) {
            // In a real library, we'd throw an exception here.
            // For this example, we'll just proceed, but the behavior will be undefined
            // if the dimensions don't match the iterator size.
        }
    }

    auto get(size_t i, size_t j, size_t k) const {
        return m_iter[i * m_dim_j * m_dim_k + j * m_dim_k + k];
    }

    size_t size() const {
        return m_iter.size();
    }

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
#endif // NUMITER_HPP
