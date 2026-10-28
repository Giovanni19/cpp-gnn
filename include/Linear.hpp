#pragma once
#include "Layer.hpp"
#include <vector>
#include <cstddef>

class Linear : public Layer{
private:
    std::size_t input_dimension;
    std::size_t output_dimension;

    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    void initialize_weights();
public:
    // constructor
    Linear(std::size_t input_dimension0,
           std::size_t output_dimension0);
    std::vector<double> forward(
        const std::vector<double>& input
    ) const override;
};