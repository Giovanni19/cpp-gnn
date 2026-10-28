#pragma once

#include "Layer.hpp"

#include <memory>
#include <vector>

class MLP
{
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add_layer(std::unique_ptr<Layer> layer);

    std::vector<double> forward(
        const std::vector<double>& input
    ) const;
};