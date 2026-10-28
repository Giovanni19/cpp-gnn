#pragma once

#include "Layer.hpp"

class ReLU : public Layer
{
public:
    std::vector<double> forward(
        const std::vector<double>& input
    ) const override;
};