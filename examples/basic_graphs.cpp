#include "Graph.hpp"
#include <iostream>

int main()
{
    Graph graph;
    std::vector<Node> nodes = {
        Node(0, {1.0, 2.0}),
        Node(1, {3.0, 4.0}),
        Node(2, {5.0, 6.0})
    };
    for (const Node& node : nodes)
    {
        graph.add_node(node);
    }
    std::vector<Edge> edges = {
        Edge(0, 1, {1.0, 2.0}),
        Edge(1, 2, {3.0, 4.0}),
        Edge(0, 2, {5.0, 6.0})
    };
    for (const Edge& edge : edges)
    {
        graph.add_edge(edge);
    }
    std:: cout << "Number of nodes in the graph: " << graph.num_nodes() << "\n Number of edges in the graph: " << graph.num_edges();
      for (int neighbor : graph.neighbors(0))
    {
        std::cout << " Neighbor: " << neighbor;
    }
    return 0;
}