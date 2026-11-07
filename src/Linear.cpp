#include "Linear.hpp"
#include <stdexcept>
#include <random>
#include <cmath>

Linear::Linear(
    std::size_t input_dimension0,
    std::size_t output_dimension0,
    unsigned seed
)
    : input_dimension(input_dimension0),
      output_dimension(output_dimension0),
      weights(
          output_dimension0,
          std::vector<double>(input_dimension0)
      ),
      bias(output_dimension0, 0.0),
      grad_weights(output_dimension0, std::vector<double>(input_dimension0, 0.0)),
      grad_bias(output_dimension0, 0.0)
{
    initialize_weights(seed);
}





std::vector<double> Linear::forward(
    const std::vector<double>& input
) const
{
    if (input.size() != input_dimension)
    {
        throw std::invalid_argument("Input dimension does not match Linear layer");
    }
    last_input = input;
    std::vector<double> output(output_dimension);
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

void Linear::initialize_weights(unsigned seed)
{
    double limit = std::sqrt(
        6.0 / (input_dimension + output_dimension)
    );

    std::mt19937 generator(seed);

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

std::vector<double> Linear::backward(
    const std::vector<double>& grad_output
)
{
    std::vector<double> grad_input(input_dimension, 0.0);

    for (std::size_t i = 0; i < output_dimension; ++i)
    {
        grad_bias[i] += grad_output[i];

        for (std::size_t j = 0; j < input_dimension; ++j)
        {
            grad_weights[i][j] += grad_output[i] * last_input[j];
            grad_input[j] += weights[i][j] * grad_output[i];
        }
    }
    return grad_input;
}

void Linear::step(double learning_rate)
{
    for (std::size_t i = 0; i < output_dimension; ++i)
    {
        bias[i] -= learning_rate * grad_bias[i];
        grad_bias[i] = 0.0;

        for (std::size_t j = 0; j < input_dimension; ++j)
        {
            weights[i][j] -= learning_rate * grad_weights[i][j];
            grad_weights[i][j] = 0.0;
        }
    }
}
