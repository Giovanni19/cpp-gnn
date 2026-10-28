#include "Linear.hpp"
#include <stdexcept>
#include <random>
#include <cmath>

Linear::Linear(
    std::size_t input_dimension0,
    std::size_t output_dimension0
)
    : input_dimension(input_dimension0),
      output_dimension(output_dimension0),
      weights(
          output_dimension0,
          std::vector<double>(input_dimension0)
      ),
      bias(output_dimension0, 0.0)
{
    initialize_weights();
}





std::vector<double> Linear::forward(
    const std::vector<double>& input
) const
{
    std::vector<double> output(output_dimension);
    if (input.size() != input_dimension)
    {
        throw std::invalid_argument("Input dimension does not match Linear layer");
    }
    for (std::size_t i = 0; i < output_dimension; ++i)
    {
        output[i] = bias[i];

        for (std::size_t j = 0; j < input_dimension; ++j)
        {
            output[i] += weights[i][j]*input[j];
        }
    }
    return output;
}

void Linear::initialize_weights()
{
    double limit = std::sqrt(
        6.0 / (input_dimension + output_dimension)
    );

    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_real_distribution<double> distribution(
        -limit,
        limit
    );
    for (auto& row : weights)
    {
        for (auto& weight : row)
        {
            weight = distribution(generator);
        }
    }
}