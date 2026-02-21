#pragma once

#include "quadratic.hpp"
#include <stdexcept>
#include <type_traits>
#include <functional>

namespace numiter {

// --- General Overloads (Fallback) ---

template <typename Lhs, typename Rhs>
HWY_INLINE auto operator+(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr(std::plus<>{}, lhs.derived(), rhs.derived());
}

template <typename Lhs, typename Rhs>
HWY_INLINE auto operator-(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr(std::minus<>{}, lhs.derived(), rhs.derived());
}

template <typename Lhs, typename Rhs>
HWY_INLINE auto operator*(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr(std::multiplies<>{}, lhs.derived(), rhs.derived());
}

// --- AP & Quadratic Optimizations ---

template <typename T>
HWY_INLINE auto operator+(const ArithmeticProgression<T>& a1, const ArithmeticProgression<T>& a2) {
    if (a1.size() != 0 && a2.size() != 0 && a1.size() != a2.size()) throw std::runtime_error("Incompatible sizes");
    return ArithmeticProgression<T>(a1.c() + a2.c(), a1.b() + a2.b(), a1.size() != 0 ? a1.size() : a2.size());
}

template <typename T>
HWY_INLINE auto operator+(const Quadratic<T>& q1, const Quadratic<T>& q2) {
    if (q1.size() != 0 && q2.size() != 0 && q1.size() != q2.size()) throw std::runtime_error("Incompatible sizes");
    return Quadratic<T>(q1.a() + q2.a(), q1.b() + q2.b(), q1.c() + q2.c(), q1.size() != 0 ? q1.size() : q2.size());
}

template <typename T>
HWY_INLINE auto operator*(const ArithmeticProgression<T>& a1, const ArithmeticProgression<T>& a2) {
    if (a1.size() != 0 && a2.size() != 0 && a1.size() != a2.size()) throw std::runtime_error("Incompatible sizes");
    size_t n = a1.size() != 0 ? a1.size() : a2.size();
    return Quadratic<T>(a1.b() * a2.b(), a1.b() * a2.c() + a2.b() * a1.c(), a1.c() * a2.c(), n);
}

// --- Scalar Overloads ---

template <typename T> concept not_numiter = !requires(T t) { t.derived(); };

template <typename S, typename T>
requires not_numiter<S> && std::is_arithmetic_v<S>
HWY_INLINE auto operator*(const S& s, const ArithmeticProgression<T>& a) {
    return ArithmeticProgression<T>(static_cast<T>(s * a.c()), static_cast<T>(s * a.b()), a.size());
}

template <typename S, typename T>
requires not_numiter<S> && std::is_arithmetic_v<S>
HWY_INLINE auto operator*(const S& s, const Quadratic<T>& q) {
    return Quadratic<T>(static_cast<T>(s * q.a()), static_cast<T>(s * q.b()), static_cast<T>(s * q.c()), q.size());
}

template <typename Lhs, typename S>
requires not_numiter<S> && std::is_arithmetic_v<S>
HWY_INLINE auto operator*(const numiterator<Lhs>& lhs, const S& rhs) {
    return rhs * lhs.derived();
}

} // namespace numiter
