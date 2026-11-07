#pragma once

#include "Layer.hpp"

class ReLU : public Layer
{
private:
    mutable std::vector<double> last_input;

public:
    std::vector<double> forward(
        const std::vector<double>& input
    ) const override;
    std::vector<double> backward(
        const std::vector<double>& grad_output
    ) override;
};