#include "MessagePassing.hpp"

#include <cassert>
#include <iostream>

// Obiettivo: ogni nodo deve diventare la somma delle feature dei nodi che puntano a lui.
int main()
{
    Graph graph;
    graph.add_node(Node(0, {1.0, 0.0}));
    graph.add_node(Node(1, {0.0, 1.0}));
    graph.add_node(Node(2, {0.5, -0.5}));
    graph.add_node(Node(3, {-1.0, 0.5}));
    graph.add_edge(Edge(0, 1, {1.0, 1.0}));
    graph.add_edge(Edge(2, 1, {1.0, 1.0}));
    graph.add_edge(Edge(1, 3, {1.0, 1.0}));
    graph.add_edge(Edge(3, 0, {1.0, 1.0}));

    std::vector<std::vector<double>> targets(graph.num_nodes(), {0.0, 0.0});
    for (const Edge& e : graph.get_edges())
    {
        for (int k = 0; k < 2; ++k)
        {
            targets[e.target][k] += graph.get_node(e.source).features[k];
        }
    }

    MessagePassing mp(2, 2, 16, 42);
    double loss = 0.0;

    for (int epoch = 0; epoch <= 2000; ++epoch)
    {
        auto output = mp.forward(graph);
        std::vector<std::vector<double>> grad(output.size(), std::vector<double>(2));
        loss = 0.0;

        for (std::size_t i = 0; i < output.size(); ++i)
        {
            for (int k = 0; k < 2; ++k)
            {
                double error = output[i][k] - targets[i][k];
                loss += error * error / output.size();
                grad[i][k] = 2.0 * error / output.size();
            }
        }
        mp.backward(graph, grad);
        mp.step(0.05);

        if (epoch % 500 == 0)
        {
            std::cout << "epoch " << epoch << "  loss " << loss << "\n";
        }
    }

    assert(loss < 0.01);
    return 0;
}
