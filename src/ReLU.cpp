#include "ReLU.hpp"

std::vector<double> ReLU::forward(
    const std::vector<double>& input
) const
{
    std::vector<double> output = input;
    for (double& value : output)
    {
        if(value<0){
            value=0;
        }
    }
    return output;
}