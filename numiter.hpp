#pragma once

#include "quadratic.hpp"
#include "operators.hpp"
#include "simd_engine.hpp"

#include <stdexcept>
#include <cmath>
#include <ranges>

namespace numiter {

// --- Concrete Iterators and Adapters ---

template <std::ranges::random_access_range R>
class RangeAdapter : public numiterator<RangeAdapter<R>> {
public:
    using value_type = std::ranges::range_value_t<R>;
    HWY_INLINE RangeAdapter(R range) : m_range(range) {}
    HWY_INLINE auto get(size_t index) const { return m_range[index]; }
    HWY_INLINE size_t size() const { return std::ranges::size(m_range); }
private:
    R m_range;
};

template <typename T>
class ArrayIterator : public numiterator<ArrayIterator<T>> {
public:
    using value_type = T;
    HWY_INLINE ArrayIterator(const T* data, size_t size, size_t stride = 1)
        : m_data(data), m_size(size), m_stride(stride) {}
    HWY_INLINE T get(size_t index) const { return m_data[index * m_stride]; }
    HWY_INLINE size_t size() const { return m_size; }
private:
    const T* m_data;
    size_t m_size;
    size_t m_stride;
};

template <typename Iter>
class View3D {
public:
    HWY_INLINE View3D(const Iter& iter, size_t dim_i, size_t dim_j, size_t dim_k)
        : m_iter(iter), m_dim_i(dim_i), m_dim_j(dim_j), m_dim_k(dim_k) {}
    HWY_INLINE auto get(size_t i, size_t j, size_t k) const {
        return m_iter[i * m_dim_j * m_dim_k + j * m_dim_k + k];
    }
    HWY_INLINE size_t size() const { return m_iter.size(); }
private:
    Iter m_iter;
    size_t m_dim_i, m_dim_j, m_dim_k;
};

// --- Factory Functions ---

template <typename T>
HWY_INLINE auto range(T start, T stop, T stride = 1) {
    if (stride == 0) throw std::invalid_argument("stride must not be zero");
    size_t count = 0;
    if ((stride > 0 && stop > start) || (stride < 0 && stop < start)) {
        double d_count = std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(stride) - 1e-10);
        if (d_count > 0) {
            count = static_cast<size_t>(d_count);
        }
    }
    return ArithmeticProgression<T>(start, stride, count);
}

template <typename T>
HWY_INLINE auto iota(T start, T stop) {
    size_t count = (stop > start) ? static_cast<size_t>(stop - start) : 0;
    return ArithmeticProgression<T>(start, static_cast<T>(1), count);
}

template <typename T>
HWY_INLINE auto from_array(const T* data, size_t size, size_t stride = 1) {
    return ArrayIterator<T>(data, size, stride);
}

// --- Reductions ---

template <typename Iter>
HWY_INLINE auto sum(const numiterator<Iter>& iter) {
    if constexpr (is_quadratic_v<Iter>) {
        return iter.derived().sum();
    } else {
        using T = typename Iter::value_type;
        T total = 0;
        for (size_t i = 0; i < iter.size(); ++i) {
            total += iter.derived()[i];
        }
        return total;
    }
}

template <typename Iter>
HWY_INLINE auto mean(const numiterator<Iter>& iter) {
    return sum(iter) / static_cast<double>(iter.size());
}

template <typename Iter>
HWY_INLINE auto product(const numiterator<Iter>& iter) {
    using T = typename Iter::value_type;
    T prod = 1;
    for (size_t i = 0; i < iter.size(); ++i) {
        prod *= iter.derived()[i];
    }
    return prod;
}

template <typename Iter>
HWY_INLINE auto min(const numiterator<Iter>& iter) {
    if (iter.size() == 0) throw std::runtime_error("min of empty iterator");
    auto m = iter.derived()[0];
    for (size_t i = 1; i < iter.size(); ++i) {
        auto val = iter.derived()[i];
        if (val < m) m = val;
    }
    return m;
}

template <typename Iter>
HWY_INLINE auto max(const numiterator<Iter>& iter) {
    if (iter.size() == 0) throw std::runtime_error("max of empty iterator");
    auto m = iter.derived()[0];
    for (size_t i = 1; i < iter.size(); ++i) {
        auto val = iter.derived()[i];
        if (val > m) m = val;
    }
    return m;
}

// --- Generic UFuncs ---

template <typename Iter> auto sin(const numiterator<Iter>& it) { return UnaryExpr([](auto x) { return std::sin(x); }, it.derived()); }
template <typename Iter> auto cos(const numiterator<Iter>& it) { return UnaryExpr([](auto x) { return std::cos(x); }, it.derived()); }
template <typename Iter> auto tan(const numiterator<Iter>& it) { return UnaryExpr([](auto x) { return std::tan(x); }, it.derived()); }
template <typename Iter> auto log(const numiterator<Iter>& it) { return UnaryExpr([](auto x) { return std::log(x); }, it.derived()); }
template <typename Iter> auto exp(const numiterator<Iter>& it) { return UnaryExpr([](auto x) { return std::exp(x); }, it.derived()); }
template <typename Iter> auto sqrt(const numiterator<Iter>& it) { return UnaryExpr([](auto x) { return std::sqrt(x); }, it.derived()); }
template <typename Iter> auto abs(const numiterator<Iter>& it) { return UnaryExpr([](auto x) { return std::abs(x); }, it.derived()); }

template <typename Lhs, typename Rhs>
auto pow(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs) {
    return BinaryExpr([](auto x, auto y) { return std::pow(x, y); }, lhs.derived(), rhs.derived());
}

template <typename Iter, typename F> auto map(const numiterator<Iter>& iter, F f) { return UnaryExpr(f, iter.derived()); }
template <typename Lhs, typename Rhs, typename F> auto zip(const numiterator<Lhs>& lhs, const numiterator<Rhs>& rhs, F f) { return BinaryExpr(f, lhs.derived(), rhs.derived()); }

} // namespace numiter
