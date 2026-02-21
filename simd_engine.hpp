#pragma once

#include "quadratic.hpp"
#include "hwy/highway.h"

namespace numiter {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename Derived>
class SIMDIterator {
public:
    using T = typename Derived::value_type;
    using D = hn::ScalableTag<T>;
    using V = hn::Vec<D>;

    HWY_INLINE SIMDIterator(const Derived& it, size_t start_index = 0)
        : m_it(it) {
        seek(start_index);
    }

    HWY_INLINE void seek(size_t n) {
        D d;
        const size_t W = hn::Lanes(d);

        V v_n = hn::Set(d, static_cast<T>(n));
        V v_iota = hn::Iota(d, 0);
        V v_i = hn::Add(v_n, v_iota);

        V v_a = hn::Set(d, static_cast<T>(m_it.a()));
        V v_b = hn::Set(d, static_cast<T>(m_it.b()));
        V v_c = hn::Set(d, static_cast<T>(m_it.c()));

        m_curr_value = hn::Add(hn::Add(hn::Mul(v_a, hn::Mul(v_i, v_i)), hn::Mul(v_b, v_i)), v_c);

        V v_W = hn::Set(d, static_cast<T>(W));
        V v_W2 = hn::Mul(v_W, v_W);

        V term1 = hn::Mul(hn::Set(d, static_cast<T>(2)), hn::Mul(v_a, hn::Mul(v_i, v_W)));
        V term2 = hn::Mul(v_a, v_W2);
        V term3 = hn::Mul(v_b, v_W);
        m_delta = hn::Add(hn::Add(term1, term2), term3);

        m_delta2 = hn::Set(d, static_cast<T>(2.0 * m_it.a() * W * W));
    }

    HWY_INLINE V next() {
        V out = m_curr_value;
        m_curr_value = hn::Add(m_curr_value, m_delta);
        m_delta = hn::Add(m_delta, m_delta2);
        return out;
    }

private:
    Derived m_it;
    V m_curr_value;
    V m_delta;
    V m_delta2;
};

template <typename Derived>
class VectorBuffer {
public:
    using T = typename Derived::value_type;
    using D = hn::ScalableTag<T>;
    using V = hn::Vec<D>;

    HWY_INLINE VectorBuffer(const Derived& it)
        : m_iter(it), m_count(0), m_size(it.size()) {
        m_W = hn::Lanes(D());
    }

    HWY_INLINE bool has_next() const {
        return m_count < m_size;
    }

    HWY_INLINE T next_scalar() {
        size_t buffer_index = m_count % m_W;
        if (buffer_index == 0) {
            m_curr_vec = m_iter.next();
            hn::StoreU(m_curr_vec, D(), m_buffer);
        }

        T val = m_buffer[buffer_index];
        m_count++;
        return val;
    }

private:
    SIMDIterator<Derived> m_iter;
    V m_curr_vec;
    size_t m_count;
    size_t m_size;
    size_t m_W;
    T m_buffer[128];
};

} // namespace HWY_NAMESPACE
} // namespace numiter
