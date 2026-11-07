#include "MessagePassing.hpp"

#include <iostream>

int main()
{
    Graph graph;
    graph.add_node(Node(0, {1.0, 0.0}));
    graph.add_node(Node(1, {0.0, 1.0}));
    graph.add_node(Node(2, {0.5, 0.5}));
    graph.add_edge(Edge(0, 1, {1.0}));
    graph.add_edge(Edge(1, 2, {2.0}));

    MessagePassing mp(2, 1, 8);

    auto messages = mp.update_edges(graph);
    auto nodes = mp.update_nodes(graph, messages);

    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        std::cout << "Node " << i << ":";
        for (double v : nodes[i]) std::cout << " " << v;
        std::cout << "\n";
    }
    return 0;
}
