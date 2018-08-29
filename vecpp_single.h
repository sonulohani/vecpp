//  Copyright 2018 Francois Chabot
//  (francois.chabot.dev@gmail.com)
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
#ifndef ABULAFIA_SINGLE_INCLUDE_H_
#define ABULAFIA_SINGLE_INCLUDE_H_
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#define VECPP_VERSION_MAJOR 0
#define VECPP_VERSION_MINOR 0
#define VECPP_VERSION_PATCH 1
#define VECPP_VERSION 001
#ifndef VECPP_NAMESPACE
#define VECPP_NAMESPACE vecpp
#endif

namespace VECPP_NAMESPACE {
template <typename T = float>
constexpr T pi = T(3.1415926535897932385);
template <typename T = float>
constexpr T half_pi = pi<T> / T(2);
template <typename T = float>
constexpr T two_pi = pi<T>* T(2);
}

namespace VECPP_NAMESPACE {
template <typename T>
class Angle {
  static_assert(std::is_floating_point<T>::value,
                "Only Angles of floating point types are allowed");
 public:
  using value_type = T;
  static constexpr Angle from_rad(const value_type&);
  static constexpr Angle from_deg(const value_type&);
  // The argument MUST be in the ]-PI, PI] range.
  static constexpr Angle from_clamped_rad(const value_type&);
  // The argument MUST be in the ]-180, 180] range.
  static constexpr Angle from_clamped_deg(const value_type&);
  constexpr value_type as_deg() const;
  constexpr value_type as_rad() const;
 private:
  value_type value_;
  // Constructs an angle from a constrained radian value.
  explicit constexpr Angle(const T&);
};
template <typename T>
constexpr Angle<T> operator-(const Angle<T>& rhs) {
  T value = rhs.as_rad();
  // Special case, we keep positive pi.
  if (value != pi<T>) {
    value = -value;
  }
  return Angle<T>::from_clamped_rad(value);
}
template <typename T>
constexpr Angle<T>& operator+=(Angle<T>& lhs, const Angle<T>& rhs) {
  T val = lhs.as_rad() + rhs.as_rad();
  // Since both lhs and rhs are in the ]-PI,PI] range, the sum is in the
  // ]-2PI-1,2PI] range, so we can make assumptions in the constraining process.
  if (val > pi<T>) {
    val -= two_pi<T>;
  } else if (val <= -pi<T>) {
    val += two_pi<T>;
  }
  lhs = Angle<T>::from_clamped_rad(val);
  return lhs;
}
template <typename T>
constexpr Angle<T> operator+(const Angle<T>& lhs, const Angle<T>& rhs) {
  auto result = lhs;
  result += rhs;
  return result;
}
template <typename T>
constexpr Angle<T>& operator-=(Angle<T>& lhs, const Angle<T>& rhs) {
  T val = lhs.as_rad() - rhs.as_rad();
  // Since both lhs and rhs are in the ]-PI,PI] range, the difference is in the
  // ]-2PI,2PI[ range, so we can make assumptions in the constraining process.
  if (val > pi<T>) {
    val -= two_pi<T>;
  } else if (val <= -pi<T>) {
    val += two_pi<T>;
  }
  lhs = Angle<T>::from_clamped_rad(val);
  return lhs;
}
template <typename T>
constexpr Angle<T> operator-(const Angle<T>& lhs, const Angle<T>& rhs) {
  auto result = lhs;
  result -= rhs;
  return result;
}
template <typename T>
constexpr Angle<T>& operator*=(Angle<T>& lhs, const T& rhs) {
  lhs = Angle<T>::from_rad(lhs.as_rad() * rhs);
  return lhs;
}
template <typename T>
constexpr Angle<T> operator*(const Angle<T>& lhs, const T& rhs) {
  auto result = lhs;
  result *= rhs;
  return result;
}
template <typename T>
constexpr Angle<T> operator*(const T& lhs, const Angle<T>& rhs) {
  return rhs * lhs;
}
template <typename T>
constexpr Angle<T>& operator/=(Angle<T>& lhs, const T& rhs) {
  lhs = Angle<T>::from_rad(lhs.as_rad() / rhs);
  return lhs;
}
template <typename T>
constexpr Angle<T> operator/(const Angle<T>& lhs, const T& rhs) {
  auto result = lhs;
  result /= rhs;
  return result;
}
template <typename T>
constexpr bool operator==(const Angle<T>& lhs, const Angle<T>& rhs) {
  return lhs.as_rad() == rhs.as_rad();
}
template <typename T>
constexpr bool operator!=(const Angle<T>& lhs, const Angle<T>& rhs) {
  return lhs.as_rad() != rhs.as_rad();
}
template <typename T>
constexpr bool operator<(const Angle<T>& lhs, const Angle<T>& rhs) {
  return lhs.as_rad() < rhs.as_rad();
}
template <typename T>
constexpr bool operator>(const Angle<T>& lhs, const Angle<T>& rhs) {
  return lhs.as_rad() > rhs.as_rad();
}
template <typename T>
constexpr bool operator<=(const Angle<T>& lhs, const Angle<T>& rhs) {
  return lhs.as_rad() <= rhs.as_rad();
}
template <typename T>
constexpr bool operator>=(const Angle<T>& lhs, const Angle<T>& rhs) {
  return lhs.as_rad() >= rhs.as_rad();
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Angle<T>& v) {
  return stream << v.as_deg() << "°";
}
template <typename T>
constexpr Angle<T>::Angle(const T& v) : value_(v) {}
template <typename T>
constexpr Angle<T> Angle<T>::from_clamped_rad(const T& v) {
  assert(v > -pi<float> && v <= pi<float>);
  return Angle<T>(v);
}
template <typename T>
constexpr Angle<T> Angle<T>::from_clamped_deg(const T& v) {
  return from_clamped_rad(v / T(180) * pi<T>);
}
template <typename T>
constexpr Angle<T> Angle<T>::from_rad(const T& v) {
  // Unfortunately, std::fmod is not constexpr, so we have to roll our own...
  T constrained = v + pi<T>;
  T div = static_cast<T>(static_cast<long long int>(constrained / two_pi<T>));
  constrained -= div * two_pi<T>;
  if (constrained <= T(0)) {
    constrained += two_pi<T>;
  }
  constrained -= pi<T>;
  return from_clamped_rad(constrained);
}
template <typename T>
constexpr Angle<T> Angle<T>::from_deg(const T& v) {
  return from_rad(v / T(180) * pi<T>);
}
template <typename T>
constexpr T Angle<T>::as_deg() const {
  return value_ * T(180) / pi<T>;
}
template <typename T>
constexpr T Angle<T>::as_rad() const {
  return value_;
}
}

namespace VECPP_NAMESPACE {
template <typename T, std::size_t len>
struct Vec {
 public:
  using value_type = T;
  constexpr std::size_t size() const { return len; }
  constexpr T& at(std::size_t i) {
    if (i >= len) {
      throw std::out_of_range("out of range vector access");
    }
    return data_[i];
  }
  constexpr const T& at(std::size_t i) const {
    if (i >= len) {
      throw std::out_of_range("out of range vector access");
    }
    return data_[i];
  }
  constexpr T& operator[](std::size_t i) {
    assert(i < len);
    return data_[i];
  }
  constexpr const T& operator[](std::size_t i) const {
    assert(i < len);
    return data_[i];
  }
  constexpr T* data() { return data_.data(); }
  constexpr const T* data() const { return data_.data(); }
  // Left public for aggregate initialization.
  std::array<T, len> data_;
};
template <typename T, std::size_t len>
constexpr T* begin(Vec<T, len>& v) {
  return v.data();
}
template <typename T, std::size_t len>
constexpr T* end(Vec<T, len>& v) {
  return v.data() + len;
}
template <typename T, std::size_t len>
constexpr const T* begin(const Vec<T, len>& v) {
  return v.data();
}
template <typename T, std::size_t len>
constexpr const T* end(const Vec<T, len>& v) {
  return v.data() + len;
}
template <typename T, std::size_t L>
std::ostream& operator<<(std::ostream& stream, const Vec<T, L>& vec) {
  stream << "(";
  bool first = true;
  for(const auto& v : vec) {
    if(!first) {
      stream << ", ";
    }
    else {
      first = false;
    }
    stream << v;
  }
  stream << ")";
  return stream;
}
template <typename T, std::size_t L>
constexpr bool operator==(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}
template <typename T, std::size_t L>
constexpr bool operator!=(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    if (lhs[i] != rhs[i]) {
      return true;
    }
  }
  return false;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator-(const Vec<T, L>& rhs) {
  Vec<T, L> result = {};
  for (std::size_t i = 0; i < L; ++i) {
    result[i] = -rhs[i];
  }
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L>& operator+=(Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    lhs[i] += rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator+(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  Vec<T, L> result = lhs;
  result += rhs;
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L>& operator-=(Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    lhs[i] -= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator-(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  Vec<T, L> result = lhs;
  result -= rhs;
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L>& operator*=(Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    lhs[i] *= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator*(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  Vec<T, L> result = lhs;
  result *= rhs;
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L>& operator/=(Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    lhs[i] /= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator/(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  Vec<T, L> result = lhs;
  result /= rhs;
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L>& operator*=(Vec<T, L>& lhs, const T& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    lhs[i] *= rhs;
  }
  return lhs;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator*(const Vec<T, L>& lhs, const T& rhs) {
  Vec<T, L> result = lhs;
  result *= rhs;
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator*(const T& lhs, const Vec<T, L>& rhs) {
  return rhs * lhs;
}
template <typename T, std::size_t L>
constexpr Vec<T, L>& operator/=(Vec<T, L>& lhs, const T& rhs) {
  for (std::size_t i = 0; i < L; ++i) {
    lhs[i] /= rhs;
  }
  return lhs;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> operator/(const Vec<T, L>& lhs, const T& rhs) {
  Vec<T, L> result = lhs;
  result /= rhs;
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> min(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  Vec<T, L> result = {0};
  for (std::size_t i = 0; i < L; ++i) {
    result[i] = std::min(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> max(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
  Vec<T, L> result = {0};
  for (std::size_t i = 0; i < L; ++i) {
    result[i] = std::max(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t L>
constexpr Vec<T, L> abs(const Vec<T, L>& vec) {
  Vec<T, L> result = {0};
  for (std::size_t i = 0; i < L; ++i) {
    result[i] = std::abs(vec[i]);
  }
  return result;
}
template <typename T>
struct Dot_impl;
template <typename T, std::size_t L>
struct Dot_impl<Vec<T, L>> {
  constexpr static T dot(const Vec<T, L>& lhs, const Vec<T, L>& rhs) {
    T result = 0;
    for (std::size_t i = 0; i < L; ++i) {
      result += lhs[i] * rhs[i];
    }
    return result;
  }
};
template <typename T>
constexpr auto dot(const T& lhs, const T& rhs) {
  return Dot_impl<T>::dot(lhs, rhs);
}
template <typename T>
struct Length_impl;
template <typename T, std::size_t L>
struct Length_impl<Vec<T, L>> {
  constexpr static T length(const Vec<T, L>& v) { return std::sqrt(dot(v, v)); }
};
template <typename T>
constexpr auto length(const T& v) {
  return Length_impl<T>::length(v);
}
template <typename T>
struct Cross_impl;
template <typename T>
struct Cross_impl<Vec<T, 3>> {
  static constexpr Vec<T, 3> cross(const Vec<T, 3>& lhs, const Vec<T, 3>& rhs) {
    return {lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[0] * rhs[1] - lhs[1] * rhs[0]};
  }
};
template <typename T>
constexpr auto cross(const T& lhs, const T& rhs) {
  return Cross_impl<T>::cross(lhs, rhs);
}
}

namespace VECPP_NAMESPACE {
template <typename T, std::size_t col_count, std::size_t row_count>
struct Mat {
  using value_type = T;
  using col_type = Vec<T, row_count>;
  constexpr col_type& operator[](std::size_t i) { return data_[i]; }
  constexpr const col_type& operator[](std::size_t i) const { return data_[i]; }
  col_type* data() { return data_.data(); }
  const col_type* data() const { return data_.data(); }
  static constexpr Mat make_identity();
  static constexpr Mat make_zero();
  // Left public for aggregate initialization.
  std::array<col_type, col_count> data_;
};
template <typename T, std::size_t C, std::size_t R>
constexpr bool operator==(const Mat<T, C, R>& lhs, const Mat<T, C, R>& rhs) {
  for (std::size_t i = 0; i < C; ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}
template <typename T, std::size_t C, std::size_t R>
constexpr bool operator!=(const Mat<T, C, R>& lhs, const Mat<T, C, R>& rhs) {
  for (std::size_t i = 0; i < C; ++i) {
    if (lhs[i] != rhs[i]) {
      return true;
    }
  }
  return false;
}
template <typename T, std::size_t C, std::size_t R>
constexpr Mat<T, C, R> Mat<T, C, R>::make_identity() {
  Mat<T, C, R> result = {0};
  for (std::size_t i = 0; i < std::min(C, R); ++i) {
    result[i][i] = T(1);
  }
  return result;
}
template <typename T, std::size_t C, std::size_t R>
constexpr Mat<T, C, R> Mat<T, C, R>::make_zero() {
  return {0};
}
}

namespace VECPP_NAMESPACE {
  template<typename T>
  constexpr T identity = T::make_identity();
  template<typename T>
  constexpr T zero = T::make_zero();
}

namespace VECPP_NAMESPACE {
template <typename T>
constexpr Mat<T, 4, 4> make_ortho(T l, T r, T b, T t) {
  Mat<T, 4, 4> result = Mat<T, 4, 4>::identity;
  result[0][0] = T(2) / (r - l);
  result[1][1] = T(2) / (t - b);
  result[3][0] = (r + l) / (r - l);
  result[3][1] = (t + b) / (t - b);
  return result;
}
}

namespace VECPP_NAMESPACE {
template <typename T, std::size_t A, std::size_t B, std::size_t C>
struct Matmul_impl {
  constexpr static Mat<T, A, C> mul(const Mat<T, A, B>& lhs,
                                    const Mat<T, B, C>& rhs) {
    Mat<T, A, C> result = {};
    for (std::size_t i = 0; i < A; ++i) {
      for (std::size_t j = 0; j < C; ++j) {
        for (std::size_t k = 0; k < B; ++k) {
          result[i][j] += lhs[i][k] * rhs[k][j];
        }
      }
    }
    return result;
  }
};
template <typename T, std::size_t A, std::size_t B, std::size_t C>
constexpr Mat<T, A, C> operator*(const Mat<T, A, B>& lhs,
                                 const Mat<T, B, C>& rhs) {
  return Matmul_impl<T, A, B, C>::mul(lhs, rhs);
}
}

namespace VECPP_NAMESPACE {
template <typename T>
struct Quat {
  using value_type = T;
  static constexpr Quat angle_axis(const Angle<T>& angle,
                                   const Vec<T, 3>& axis);
  // Left public for aggregate initialization.
  T w;
  T x;
  T y;
  T z;
};
template <typename T>
constexpr Quat<T> Quat<T>::angle_axis(const Angle<T>& angle,
                                      const Vec<T, 3>& axis) {
  const T s = std::sin(angle.as_rad() * T(0.5));
  const T c = std::cos(angle.as_rad() * T(0.5));
  return {c, axis[0] * s, axis[1] * s, axis[2] * s};
}
template <typename T>
constexpr Quat<T>& operator*=(Quat<T>& lhs, const Quat<T>& rhs) {
  const Quat<T> p(lhs);
  const Quat<T> q(rhs);
  lhs.w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
  lhs.x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
  lhs.y = p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z;
  lhs.z = p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x;
  return lhs;
}
template <typename T>
constexpr Quat<T> operator*(const Quat<T>& lhs, const Quat<T>& rhs) {
  Quat<T> result(lhs);
  result *= rhs;
  return result;
}
template <typename T>
constexpr Vec<T, 3> operator*(const Quat<T>& lhs, const Vec<T, 3>& rhs) {
  const Vec<T, 3> q_v = {lhs.x, lhs.y, lhs.z};
  const Vec<T, 3> uv = cross(q_v, rhs);
  const Vec<T, 3> uuv = cross(q_v, uv);
  return rhs + ((uv * lhs.w) + uuv) * T(2);
}
}

namespace VECPP_NAMESPACE {
template <typename T>
struct Scale {
  constexpr Scale(T const& s) : scale_({s, s, s}) {}
  constexpr Scale(Vec<T, 3> const& s) : scale_(s) {}
  Vec<T, 3> scale_;
  explicit constexpr operator Mat<T, 4, 4>() {
    Mat<T, 4, 4> result = Mat<T, 4, 4>::identity;
    result[0][0] = scale_[0];
    result[1][1] = scale_[1];
    result[2][2] = scale_[2];
    return result;
  }
  constexpr Mat<T, 4, 4> right_mul(const Mat<T, 4, 4>& m) const {
    Mat<T, 4, 4> result = {};
    result[0] = m[0] * scale_[0];
    result[1] = m[1] * scale_[1];
    result[2] = m[2] * scale_[2];
    result[3] = m[3];
    return result;
  }
};
template <typename T>
constexpr Scale<T> scale(const T& s) {
  return {s};
}
template <typename T>
constexpr Scale<T> scale(const Vec<T, 3>& s) {
  return {s};
}
template <typename T>
constexpr Mat<T, 4, 4> operator*(const Mat<T, 4, 4>& m, const Scale<T>& s) {
  return s.right_mul(m);
}
template <typename T>
constexpr Vec<T, 3> operator*(const Scale<T>& s, const Vec<T, 3>& v) {
  return {v[0] * s.scale_[0], v[1] * s.scale_[1], v[2] * s.scale_[2]};
}
}


#endif
