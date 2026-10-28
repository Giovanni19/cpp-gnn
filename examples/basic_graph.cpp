#include "ReLU.hpp"

#include <iostream>
#include <vector>

int main()
{
    ReLU relu;

    std::vector<double> input = {
        -2.0,
        1.5,
        -0.3,
        4.0
    };

    std::vector<double> output = relu.forward(input);

    std::cout << "Input:  ";

    for (double value : input)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";

    std::cout << "Output: ";

    for (double value : output)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";

    return 0;
}