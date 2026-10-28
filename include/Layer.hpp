#pragma once

#include <vector>

class Layer
{
public:
    virtual std::vector<double> forward(
        const std::vector<double>& input
    ) const = 0;

    virtual ~Layer() = default;
};