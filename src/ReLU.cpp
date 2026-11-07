#include "ReLU.hpp"

std::vector<double> ReLU::forward(
    const std::vector<double>& input
) const
{
    last_input = input;
    std::vector<double> output = input;
    for (double& value : output)
    {
        if(value<0){
            value=0;
        }
    }
    return output;
}

std::vector<double> ReLU::backward(
    const std::vector<double>& grad_output
)
{
    std::vector<double> grad_input = grad_output;
    for (std::size_t i = 0; i < grad_input.size(); ++i)
    {
        if (last_input[i] <= 0)
        {
            grad_input[i] = 0;
        }
    }
    return grad_input;
}
