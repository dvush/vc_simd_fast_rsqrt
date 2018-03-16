#include <Vc/Vc>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
using Vc::float_v;
using Vc::double_v;

double_v fast_rsqrt(double_v x) {
  double_v r = (double_v)Vc::rsqrt((float_v)x);
  r = (r * (3.0 * 0.5 - (0.5 * x) * r * r)); // Newton step
  r = (r * (3.0 * 0.5 - (0.5 * x) * r * r)); // Newton step
  return r;
}

void fast_rsqrt2(double_v x1, double_v x2, double_v &r1, double_v &r2) {
  float_v f1 = (float_v)Vc::rsqrt((float_v)x1);
  float_v f2 = (float_v)Vc::rsqrt((float_v)x2);
  r1 = (double_v)(f1);
  r1 = (r1 * (3.0 * 0.5 - 0.5 * x1 * r1 * r1)); // Newton step
  r1 = (r1 * (3.0 * 0.5 - 0.5 * x1 * r1 * r1)); // Newton step

  r2 = (double_v)(f2);
  r2 = (r2 * (3.0 * 0.5 - 0.5 * x2 * r2 * r2)); // Newton step
  r2 = (r2 * (3.0 * 0.5 - 0.5 * x2 * r2 * r2)); // Newton step
}

void genearateData(std::vector<double> &data, int N) {
  std::default_random_engine eng;
  std::uniform_real_distribution<double> d(0.0, 100.0);
  data.clear();
  for (int i = 0; i < N; ++i) {
    data.push_back(d(eng));
  }
}

void checkErrors() {
  std::cout << "Test of correctness: " << std::endl;
  const int N = 1024 * 10000;
  double cum_error_1 = 0.0;
  double max_error_1 = 0.0;

  double cum_error_2 = 0.0;
  double max_error_2 = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  for (int i = 0; i < N; i += double_v::size() * 2) {
    double_v x1;
    x1.load(&data.data()[i], Vc::Unaligned);
    double_v x2;
    x2.load(&data.data()[i + double_v::size()], Vc::Unaligned);

    double_v r1_1, r2_1;
    r1_1 = fast_rsqrt(x1);
    r2_1 = fast_rsqrt(x2);

    double_v r1_2, r2_2;
    fast_rsqrt2(x1, x2, r1_2, r2_2);

    double_v r1_full = 1.0 / Vc::sqrt(x1);
    double_v r2_full = 1.0 / Vc::sqrt(x2);

    double_v error_1 = Vc::abs(r1_1 - r1_full) + Vc::abs(r2_1 - r2_full);
    double_v error_2 = Vc::abs(r1_2 - r1_full) + Vc::abs(r2_2 - r2_full);
    max_error_1 = std::max(max_error_1, error_1.max());
    max_error_2 = std::max(max_error_2, error_2.max());
    cum_error_1 = error_1.sum();
    cum_error_2 = error_2.sum();
  }
  std::cout << "Trials: " << N << std::endl;
  std::cout << "Cumulative error 1: " << cum_error_1 << std::endl;
  std::cout << "Maximum error    1: " << max_error_1 << std::endl;
  std::cout << "Cumulative error 2: " << cum_error_2 << std::endl;
  std::cout << "Maximum error    2: " << max_error_2 << std::endl;
  std::cout << std::endl;
}

void benchmark() {
  std::cout << "Test of speed: " << std::endl;
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
    double_v x2;
    x2.load(&data.data()[i + double_v::size()], Vc::Unaligned);
    double_v x3;
    x3.load(&data.data()[i + 2 * double_v::size()], Vc::Unaligned);
    double_v x4;
    x4.load(&data.data()[i + 3 * double_v::size()], Vc::Unaligned);
    double_v r1, r2, r3, r4;

    r1 = fast_rsqrt(x1);
    r2 = fast_rsqrt(x2);
    r3 = fast_rsqrt(x3);
    r4 = fast_rsqrt(x3);

    sum += r1.sum() + r2.sum() + r3.sum() + r4.sum();
  }

  auto dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_rsqrt1 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Fast rsqrt1: " << musec_rsqrt1 << " (musec)"
            << " sum " << sum << std::endl;

  t1 = std::chrono::high_resolution_clock::now();

  sum = 0.0;
  for (int i = 0; i < N; i += double_v::size() * 2) {
    double_v x1;
    x1.load(&data.data()[i], Vc::Unaligned);
    double_v x2;
    x2.load(&data.data()[i + double_v::size()], Vc::Unaligned);
    double_v r1, r2;

    fast_rsqrt2(x1, x2, r1, r2);

    sum += r1.sum() + r2.sum();
  }

  dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_rsqrt2 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Fast rsqrt2: " << musec_rsqrt2 << " (musec)"
            << " sum " << sum << std::endl;

  t1 = std::chrono::high_resolution_clock::now();

  sum = 0.0;
  for (int i = 0; i < N; i += double_v::size() * 2) {
    double_v x1;
    x1.load(&data.data()[i], Vc::Unaligned);
    double_v x2;
    x2.load(&data.data()[i + double_v::size()], Vc::Unaligned);
    double_v r1, r2;

    r1 = 1.0 / Vc::sqrt(x1);
    r2 = 1.0 / Vc::sqrt(x2);

    sum += r1.sum() + r2.sum();
  }

  dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec_full =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Full rsqrt:  " << musec_full << " (musec)"
            << " sum " << sum << std::endl;

  std::cout << "Fast rsqrt1 speedup: " << (double)musec_full / musec_rsqrt1
            << std::endl;
  std::cout << "Fast rsqrt2 speedup: " << (double)musec_full / musec_rsqrt2
            << std::endl;
}

int main() {
  checkErrors();
  benchmark();

  return 0;
}