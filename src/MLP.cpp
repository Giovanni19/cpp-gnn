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

std::vector<double> MLP::backward(const std::vector<double>& grad_output)
{
    std::vector<double> grad = grad_output;

    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }
    return grad;
}

void MLP::step(double learning_rate)
{
    for (auto& layer : layers)
    {
        layer->step(learning_rate);
    }
}
