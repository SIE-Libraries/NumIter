#pragma once

#include <cstddef>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include "hwy/highway.h"

namespace numiter {

template <typename Derived>
class numiterator {
public:
    HWY_INLINE const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }
    HWY_INLINE auto operator[](size_t index) const {
        return derived().get(index);
    }
    HWY_INLINE size_t size() const {
        return derived().size();
    }
};

template <typename T>
class ArithmeticProgression : public numiterator<ArithmeticProgression<T>> {
public:
    using value_type = T;
    HWY_INLINE ArithmeticProgression(T start, T stride, size_t size)
        : m_start(start), m_stride(stride), m_size(size) {}

    HWY_INLINE T get(size_t i) const { return static_cast<T>(m_start + i * m_stride); }
    HWY_INLINE size_t size() const { return m_size; }

    HWY_INLINE T a() const { return 0; }
    HWY_INLINE T b() const { return m_stride; }
    HWY_INLINE T c() const { return m_start; }

    HWY_INLINE T sum() const {
        if (m_size == 0) return static_cast<T>(0);
        double n = static_cast<double>(m_size);
        double B = static_cast<double>(m_stride);
        double C = static_cast<double>(m_start);
        return static_cast<T>(B * (n - 1.0) * n / 2.0 + C * n);
    }

private:
    T m_start, m_stride;
    size_t m_size;
};

template <typename T>
class Quadratic : public numiterator<Quadratic<T>> {
public:
    using value_type = T;

    HWY_INLINE Quadratic(T a, T b, T c, size_t size)
        : m_a(a), m_b(b), m_c(c), m_size(size) {}

    HWY_INLINE T get(size_t i) const {
        return static_cast<T>(m_a * i * i + m_b * i + m_c);
    }

    HWY_INLINE size_t size() const { return m_size; }
    HWY_INLINE T a() const { return m_a; }
    HWY_INLINE T b() const { return m_b; }
    HWY_INLINE T c() const { return m_c; }

    HWY_INLINE T sum() const {
        if (m_size == 0) return static_cast<T>(0);
        double n = static_cast<double>(m_size);
        double A = static_cast<double>(m_a);
        double B = static_cast<double>(m_b);
        double C = static_cast<double>(m_c);

        double result = A * (n - 1.0) * n * (2.0 * n - 1.0) / 6.0 +
                        B * (n - 1.0) * n / 2.0 +
                        C * n;
        return static_cast<T>(result);
    }

private:
    T m_a, m_b, m_c;
    size_t m_size;
};

template <typename T> struct is_quadratic : std::false_type {};
template <typename T> struct is_quadratic<Quadratic<T>> : std::true_type {};
template <typename T> struct is_ap : std::false_type {};
template <typename T> struct is_ap<ArithmeticProgression<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_quadratic_v = is_quadratic<T>::value || is_ap<T>::value;

template <typename T>
inline constexpr bool is_ap_v = is_ap<T>::value;

// --- Expression Templates ---

template <typename T>
class Scalar : public numiterator<Scalar<T>> {
public:
    using value_type = T;
    HWY_INLINE Scalar(T value) : m_value(value) {}
    HWY_INLINE T get(size_t /*index*/) const { return m_value; }
    HWY_INLINE size_t size() const { return 0; }
private:
    T m_value;
};

template <typename Op, typename Iter>
class UnaryExpr : public numiterator<UnaryExpr<Op, Iter>> {
public:
    using value_type = decltype(std::declval<Op>()(std::declval<typename Iter::value_type>()));
    HWY_INLINE UnaryExpr(Op op, const Iter& iter) : m_op(op), m_iter(iter) {}
    HWY_INLINE auto get(size_t index) const { return m_op(m_iter[index]); }
    HWY_INLINE size_t size() const { return m_iter.size(); }
private:
    Op m_op;
    Iter m_iter;
};

template <typename Op, typename Lhs, typename Rhs>
class BinaryExpr : public numiterator<BinaryExpr<Op, Lhs, Rhs>> {
public:
    using value_type = decltype(std::declval<Op>()(std::declval<typename Lhs::value_type>(), std::declval<typename Rhs::value_type>()));
    HWY_INLINE BinaryExpr(Op op, const Lhs& lhs, const Rhs& rhs) : m_op(op), m_lhs(lhs), m_rhs(rhs) {
        if (lhs.size() != 0 && rhs.size() != 0 && lhs.size() != rhs.size()) {
            throw std::runtime_error("Incompatible sizes in BinaryExpr");
        }
    }
    HWY_INLINE auto get(size_t index) const { return m_op(m_lhs[index], m_rhs[index]); }
    HWY_INLINE size_t size() const {
        if (m_lhs.size() != 0) return m_lhs.size();
        return m_rhs.size();
    }
private:
    Op m_op;
    Lhs m_lhs;
    Rhs m_rhs;
};

} // namespace numiter
