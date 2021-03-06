//  Copyright 2018 Francois Chabot
//  (francois.chabot.dev@gmail.com)
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef VECPP_VEC_OPERATIONS_PER_MEMBER_H_INCLUDED
#define VECPP_VEC_OPERATIONS_PER_MEMBER_H_INCLUDED

#include "vecpp/config.h"

#include "vecpp/scalar/operations.h"
#include "vecpp/vec/vec.h"

#include <algorithm>

namespace VECPP_NAMESPACE {

// *************** Unary functions *************** //

// abs
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> abs(const Vec<T, l, f>& vec) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < vec.size(); ++i) {
    result[i] = abs<f>(vec[i]);
  }
  return result;
}

// ceil
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> ceil(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = ceil<f>(v[i]);
  }
  return result;
}

// floor
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> floor(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = floor<f>(v[i]);
  }
  return result;
}

// fract
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> fract(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = fract<f>(v[i]);
  }
  return result;
}

// round
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> round(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = round<f>(v[i]);
  }
  return result;
}

// sign
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> sign(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = sign<f>(v[i]);
  }
  return result;
}

// trunc
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> trunc(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = trunc<f>(v[i]);
  }
  return result;
}
// *************** Binary functions *************** //

// max
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> max(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = max(lhs[i], rhs[i]);
  }
  return result;
}

// min
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> min(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = min(lhs[i], rhs[i]);
  }
  return result;
}

// mod
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> mod(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = mod<f>(lhs[i], rhs[i]);
  }
  return result;
}

// step
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> step(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = step<f>(lhs[i], rhs[i]);
  }
  return result;
}

// *************** Other functions *************** //

// clamp
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> clamp(const Vec<T, l, f>& v, const Vec<T, l, f>& low,
                             const Vec<T, l, f>& high) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = clamp<f>(v[i], low[i], high[i]);
  }
  return result;
}

// lerp
// Implicitely handled via the scalar implementation

}  // namespace VECPP_NAMESPACE
#endif