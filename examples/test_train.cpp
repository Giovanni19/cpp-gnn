#include "MLP.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

// Impara lo XOR con un MLP 2 -> 8 -> 1 e loss MSE.
int main()
{
    MLP mlp;
    mlp.add_layer(std::make_unique<Linear>(2, 8, 42));
    mlp.add_layer(std::make_unique<ReLU>());
    mlp.add_layer(std::make_unique<Linear>(8, 1, 43));

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> targets = {0, 1, 1, 0};

    double loss = 0.0;
    for (int epoch = 0; epoch <= 2000; ++epoch)
    {
        loss = 0.0;
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            double error = mlp.forward(inputs[i])[0] - targets[i];
            loss += error * error / inputs.size();
            mlp.backward({2.0 * error / inputs.size()});
        }
        mlp.step(0.1);

        if (epoch % 500 == 0)
        {
            std::cout << "epoch " << epoch << "  loss " << loss << "\n";
        }
    }

    for (const auto& x : inputs)
    {
        std::cout << x[0] << " xor " << x[1] << " -> " << mlp.forward(x)[0] << "\n";
    }

    assert(loss < 0.01);
    return 0;
}
