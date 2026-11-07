#pragma once
#include "Layer.hpp"
#include <vector>
#include <cstddef>
#include <random>

class Linear : public Layer{
private:
    std::size_t input_dimension;
    std::size_t output_dimension;

    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    std::vector<std::vector<double>> grad_weights;
    std::vector<double> grad_bias;
    mutable std::vector<double> last_input;

    void initialize_weights(unsigned seed);
public:
    // constructor
    Linear(std::size_t input_dimension0,
           std::size_t output_dimension0,
           unsigned seed = std::random_device{}());
    std::vector<double> forward(
        const std::vector<double>& input
    ) const override;
    std::vector<double> backward(
        const std::vector<double>& grad_output
    ) override;
    void step(double learning_rate) override;
};