#include "MessagePassing.hpp"

#include <cassert>
#include <iostream>

int main()
{
    Graph graph;
    graph.add_node(Node(0, {1.0, 0.0}));
    graph.add_node(Node(1, {0.0, 1.0}));
    graph.add_edge(Edge(0, 1, {1.0}));

    MessagePassing a(2, 1, 8, 42), b(2, 1, 8, 42), c(2, 1, 8, 7);

    auto ra = a.update_nodes(graph, a.update_edges(graph));
    auto rb = b.update_nodes(graph, b.update_edges(graph));
    auto rc = c.update_nodes(graph, c.update_edges(graph));

    assert(ra.size() == 2 && ra[0].size() == 2);  // una riga per nodo, stessa dimensione
    assert(ra == rb);                             // stesso seed, stesso risultato
    assert(ra != rc);                             // seed diverso, risultato diverso

    std::cout << "OK\n";
    return 0;
}
