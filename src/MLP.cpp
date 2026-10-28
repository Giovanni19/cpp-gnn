#include "MLP.hpp"

#include <utility>

void MLP::add_layer(std::unique_ptr<Layer> layer)
{
    layers.push_back(std::move(layer));
}

std::vector<double> MLP::forward(
    const std::vector<double>& input
) const
{
    std::vector<double> output = input;

    for (const auto& layer : layers)
    {
        output = layer->forward(output);
    }

    return output;
}