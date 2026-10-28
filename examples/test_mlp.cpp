#include "MLP.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main()
{
    MLP mlp;

    mlp.add_layer(std::make_unique<Linear>(3, 500));
    mlp.add_layer(std::make_unique<ReLU>());
    mlp.add_layer(std::make_unique<Linear>(500, 2));

    std::vector<double> input = {1.0, 2.0, 3.0};

    std::vector<double> output = mlp.forward(input);

    std::cout << "Input: ";

    for (double value : input)
    {
        std::cout << value << " ";
    }

    std::cout << "\nOutput: ";

    for (double value : output)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";

    return 0;
}