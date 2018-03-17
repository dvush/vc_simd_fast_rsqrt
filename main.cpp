#include <Vc/Vc>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <x86intrin.h>

using Vc::float_v;
using Vc::double_v;

double fast_rsqrt(double x) {
  float fx = (float)x;
  __m128 tmp = _mm_load_ss(&fx);
  tmp = _mm_rsqrt_ss(tmp);
  _mm_store_ss(&fx, tmp);
  double r = (double)fx;

  r = (r * (3.0 * 0.5 - (0.5 * x) * r * r)); // Newton step
  r = (r * (3.0 * 0.5 - (0.5 * x) * r * r)); // Newton step
  return r;
}

double_v fast_rsqrt(double_v x) {
  double_v r = (double_v)Vc::rsqrt((float_v)x);
  r = (r * (3.0 * 0.5 - (0.5 * x) * r * r)); // Newton step
  r = (r * (3.0 * 0.5 - (0.5 * x) * r * r)); // Newton step
  return r;
}

void genearateData(std::vector<double> &data, int N) {
  std::default_random_engine eng;
  std::uniform_real_distribution<double> d(0.0, 1000.0);
  data.clear();
  for (int i = 0; i < N; ++i) {
    data.push_back(d(eng));
  }
}

void checkErrors_vector() {
  std::cout << "Test of correctness: " << std::endl;
  const int N = 1024 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  std::vector<double> error_accum;

  for (int i = 0; i < N; i += double_v::size()) {
    double_v x;
    x.load(&data.data()[i], Vc::Unaligned);

    double_v r = fast_rsqrt(x);
    double_v r_full = 1.0 / Vc::sqrt(x);

    double_v error = Vc::abs(r - r_full) / r_full;
    max_error = std::max(max_error, error.max());
    error_accum.push_back(error.sum());
  }

  std::sort(error_accum.begin(), error_accum.end());
  cum_error = std::accumulate(error_accum.begin(), error_accum.end(),
                              0.0); // Could be replaced by better summation

  std::cout << "Trials: " << N << std::endl;
  std::cout << "Cumulative error(rel) : " << cum_error << std::endl;
  std::cout << "Average error(rel)    : " << cum_error / error_accum.size()
            << std::endl;
  std::cout << "Maximum error(rel)    : " << max_error << std::endl;
}

void checkErrors_scalar() {
  std::cout << "Test of correctness: " << std::endl;
  const int N = 1024 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  std::vector<double> error_accum;

  for (int i = 0; i < N; ++i) {
    double x = data[i];
    double r = fast_rsqrt(x);
    double r_full = 1.0 / std::sqrt(x);

    double error = std::abs(r - r_full) / r_full;
    max_error = std::max(max_error, error);
    error_accum.push_back(error);
  }

  std::sort(error_accum.begin(), error_accum.end());
  cum_error = std::accumulate(error_accum.begin(), error_accum.end(),
                              0.0); // Could be replaced by better summation

  std::cout << "Trials: " << N << std::endl;
  std::cout << "Cumulative error(rel) : " << cum_error << std::endl;
  std::cout << "Average error(rel)    : " << cum_error / error_accum.size()
            << std::endl;
  std::cout << "Maximum error(rel)    : " << max_error << std::endl;
}

void benchmark_unroll_vector() {
  std::cout << "Test of speed for pipelined execution: " << std::endl;
  const int N = 1024 * 4 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  auto t1 = std::chrono::high_resolution_clock::now();

  double sum = 0.0;
  for (int i = 0; i < N; i += double_v::size() * 4) {
    double_v x1;
    x1.load(&data.data()[i], Vc::Unaligned);
    double_v r1 = fast_rsqrt(x1);

    double_v x2;
    x2.load(&data.data()[i + double_v::size()], Vc::Unaligned);
    double_v r2 = fast_rsqrt(x2);

    double_v x3;
    x3.load(&data.data()[i + double_v::size() * 2], Vc::Unaligned);
    double_v r3 = fast_rsqrt(x3);

    double_v x4;
    x4.load(&data.data()[i + double_v::size() * 3], Vc::Unaligned);
    double_v r4 = fast_rsqrt(x4);

    sum += r1.sum() + r2.sum() + r3.sum() + r4.sum();
  }

  auto dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_rsqrt1 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Fast rsqrt : " << musec_rsqrt1 << " (musec)"
            << " sum " << sum << std::endl;

  t1 = std::chrono::high_resolution_clock::now();

  sum = 0.0;
  for (int i = 0; i < N; i += double_v::size() * 4) {
    double_v x1;
    x1.load(&data.data()[i], Vc::Unaligned);
    double_v r1 = 1.0 / Vc::sqrt(x1);

    double_v x2;
    x2.load(&data.data()[i + double_v::size() * 1], Vc::Unaligned);
    double_v r2 = 1.0 / Vc::sqrt(x2);

    double_v x3;
    x3.load(&data.data()[i + double_v::size() * 2], Vc::Unaligned);
    double_v r3 = 1.0 / Vc::sqrt(x3);

    double_v x4;
    x4.load(&data.data()[i + double_v::size() * 3], Vc::Unaligned);
    double_v r4 = 1.0 / Vc::sqrt(x4);

    sum += r1.sum() + r2.sum() + r3.sum() + r4.sum();
  }

  dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_full =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Full rsqrt:  " << musec_full << " (musec)"
            << " sum " << sum << std::endl;

  std::cout << "Fast rsqrt speedup: " << (double)musec_full / musec_rsqrt1
            << std::endl;
}

void benchmark_unroll_scalar() {
  std::cout << "Test of speed for pipelined execution: " << std::endl;
  const int N = 1024 * 4 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  auto t1 = std::chrono::high_resolution_clock::now();

  double sum = 0.0;
  for (int i = 0; i < N; i += 4) {
    double x1 = data[i];
    double r1 = fast_rsqrt(x1);

    double x2 = data[i + 1];
    double r2 = fast_rsqrt(x2);

    double x3 = data[i + 2];
    double r3 = fast_rsqrt(x3);

    double x4 = data[i + 3];
    double r4 = fast_rsqrt(x4);

    sum += r1 + r2 + r3 + r4;
  }

  auto dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_rsqrt1 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Fast rsqrt : " << musec_rsqrt1 << " (musec)"
            << " sum " << sum << std::endl;

  t1 = std::chrono::high_resolution_clock::now();

  sum = 0.0;
  for (int i = 0; i < N; i += 4) {
    double x1 = data[i];
    double r1 = 1.0 / std::sqrt(x1);

    double x2 = data[i + 1];
    double r2 = 1.0 / std::sqrt(x2);

    double x3 = data[i + 2];
    double r3 = 1.0 / std::sqrt(x3);

    double x4 = data[i + 3];
    double r4 = 1.0 / std::sqrt(x4);

    sum += r1 + r2 + r3 + r4;
  }

  dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_full =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Full rsqrt:  " << musec_full << " (musec)"
            << " sum " << sum << std::endl;

  std::cout << "Fast rsqrt speedup: " << (double)musec_full / musec_rsqrt1
            << std::endl;
}

void benchmark_vector() {
  std::cout << "Test of speed: " << std::endl;
  const int N = 1024 * 4 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  auto t1 = std::chrono::high_resolution_clock::now();

  double sum = 0.0;
  for (int i = 0; i < N; i += double_v::size() * 1) {
    double_v x1;
    x1.load(&data.data()[i], Vc::Unaligned);

    double_v r1 = fast_rsqrt(x1);

    sum += r1.sum();
  }

  auto dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_rsqrt1 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Fast rsqrt:  " << musec_rsqrt1 << " (musec)"
            << " sum " << sum << std::endl;

  t1 = std::chrono::high_resolution_clock::now();

  sum = 0.0;
  for (int i = 0; i < N; i += double_v::size() * 1) {
    double_v x1;
    x1.load(&data.data()[i], Vc::Unaligned);
    double_v r1 = 1.0 / Vc::sqrt(x1);

    sum += r1.sum();
  }

  dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_full =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Full rsqrt:  " << musec_full << " (musec)"
            << " sum " << sum << std::endl;

  std::cout << "Fast rsqrt speedup: " << (double)musec_full / musec_rsqrt1
            << std::endl;
}

void benchmark_scalar() {
  std::cout << "Test of speed: " << std::endl;
  const int N = 1024 * 4 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  auto t1 = std::chrono::high_resolution_clock::now();

  double sum = 0.0;
  for (int i = 0; i < N; ++i) {
    double x = data[i];
    double r = fast_rsqrt(x);

    sum += r;
  }

  auto dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_rsqrt1 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Fast rsqrt:  " << musec_rsqrt1 << " (musec)"
            << " sum " << sum << std::endl;

  t1 = std::chrono::high_resolution_clock::now();

  sum = 0.0;
  for (int i = 0; i < N; ++i) {
    double x = data[i];
    double r = 1.0 / std::sqrt(x);

    sum += r;
  }

  dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_full =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Full rsqrt:  " << musec_full << " (musec)"
            << " sum " << sum << std::endl;

  std::cout << "Fast rsqrt speedup: " << (double)musec_full / musec_rsqrt1
            << std::endl;
}

int main() {
  std::cout << std::setprecision(20);

  std::cout << "Checking scalar version" << std::endl << std::endl;
  checkErrors_scalar();
  std::cout << std::endl;
  benchmark_scalar();
  std::cout << std::endl;
  benchmark_unroll_scalar();
  std::cout << std::endl
            << std::endl
            << "=======================" << std::endl
            << std::endl
            << std::endl;

  std::cout << "Checking vector version" << std::endl << std::endl;
  checkErrors_vector();
  std::cout << std::endl;
  benchmark_vector();
  std::cout << std::endl;
  benchmark_unroll_vector();

  return 0;
}