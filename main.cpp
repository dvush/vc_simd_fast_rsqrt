#include <Vc/Vc>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
using Vc::float_v;
using Vc::double_v;

double_v fast_rsqrt(double_v x) {
  double_v r = (double_v)Vc::rsqrt((float_v)x);
  r = 0.5 * (r * (3.0 - x * r * r)); // Newton step
  r = 0.5 * (r * (3.0 - x * r * r)); // Newton step
  return r;
}

void genearateData(std::vector<double> &data, int N) {
  std::minstd_rand eng;
  std::uniform_real_distribution<double> d(0.0, 100.0);
  data.clear();
  for (int i = 0; i < N; ++i) {
    data.push_back(d(eng));
  }
}

void checkErrors() {
  std::cout << "Test of correctness: " << std::endl;
  const int N = 1024 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  for (int i = 0; i < N; i += double_v::size()) {
    double_v x = 0.26;
    x.load(&data.data()[i], Vc::Unaligned);
    double_v r1 = fast_rsqrt(x);
    double_v r2 = 1.0 / Vc::sqrt(x);

    double_v error = Vc::abs(r1 - r2);
    max_error = std::max(max_error, error.max());
    cum_error = error.sum();
  }
  std::cout << "Trials: " << N << std::endl;
  std::cout << "Cumulative error: " << cum_error << std::endl;
  std::cout << "Maximum error   : " << max_error << std::endl;
  std::cout << std::endl;
}

void benchmark() {
  std::cout << "Test of speed: " << std::endl;
  const int N = 1024 * 10000;
  double cum_error = 0.0;
  double max_error = 0.0;

  std::vector<double> data;
  genearateData(data, N);

  auto t1 = std::chrono::high_resolution_clock::now();

  double sum = 0.0;
  for (int i = 0; i < N; i += double_v::size()) {
    double_v x;
    x.load(&data.data()[i], Vc::Unaligned);
    double_v r = fast_rsqrt(x);
    sum += r.sum();
  }

  auto dt1 = std::chrono::high_resolution_clock::now() - t1;
  auto musec1 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt1).count();

  std::cout << "Fast rsqrt: " << musec1 << " (musec)"
            << " sum " << sum << std::endl;

  auto t2 = std::chrono::high_resolution_clock::now();

  sum = 0.0;
  for (int i = 0; i < N; i += double_v::size()) {
    double_v x;
    x.load(&data.data()[i], Vc::Unaligned);
    double_v r = 1.0 / Vc::sqrt(x);
    sum += r.sum();
  }

  auto dt2 = std::chrono::high_resolution_clock::now() - t2;
  auto musec2 =
      std::chrono::duration_cast<std::chrono::microseconds>(dt2).count();

  std::cout << "Full rsqrt: " << musec2 << " (musec)"
            << " sum " << sum << std::endl;

  std::cout << "Fast rsqrt speedup: " << (double)musec2 / musec1 << std::endl;
}

int main() {
  checkErrors();
  benchmark();

  return 0;
}