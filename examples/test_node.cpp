#include "Node.hpp"
#include <iostream>

int main()
{
    Node node(0, {1.2, 0.5, -0.3});

    std::cout << "Node ID: " << node.id << "\n";

    for (double feature : node.features)
    {
        std::cout << feature << " ";
    }

    return 0;
}