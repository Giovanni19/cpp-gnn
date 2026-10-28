#pragma once

#include "Node.hpp"
#include "Edge.hpp"

#include <vector>

class Graph{
    private: 
    std:: vector<Node> nodes;
    std:: vector<Edge> edges;
    public:
    void add_node(const Node& node);
    void add_edge(const Edge& edge);
    std::size_t num_nodes() const;
    std::size_t num_edges() const;
    std::vector<int> neighbors(int node_id) const;
    const Node& get_node(int node_id) const;
    const std::vector<Edge>& get_edges() const;
};