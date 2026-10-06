/*
 * Minimal stand-in for the ROS 1 `random_numbers` package, which has no
 * ROS 2 release. Only the subset of the API that msckf_vio uses is
 * provided, with the same inclusive-bounds semantics.
 */

#ifndef MSCKF_VIO_RANDOM_NUMBERS_H
#define MSCKF_VIO_RANDOM_NUMBERS_H

#include <random>
#include <ctime>
#include <stdexcept>

namespace random_numbers {

class RandomNumberGenerator {
public:
  RandomNumberGenerator() : generator_(1) {}

  explicit RandomNumberGenerator(unsigned int seed) : generator_(seed) {}

  /*
   * @brief uniformInteger Draw a uniformly distributed integer in the
   *    closed interval [min, max], i.e. both bounds can be returned.
   *    As in the original package, the bounds are swapped if the caller
   *    passes them the other way round.
   */
  int uniformInteger(int min, int max) {
    if (min > max) {
      int temp = min;
      min = max;
      max = temp;
    }
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator_);
  }

  /*
   * @brief uniformReal Draw a uniformly distributed double in [min, max).
   */
  double uniformReal(double min, double max) {
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generator_);
  }

  /*
   * @brief gaussian Draw from a normal distribution with the given
   *    mean and standard deviation.
   */
  double gaussian(double mean, double std_dev) {
    std::normal_distribution<double> distribution(mean, std_dev);
    return distribution(generator_);
  }

private:
  std::mt19937 generator_;
};

} // namespace random_numbers

#endif
