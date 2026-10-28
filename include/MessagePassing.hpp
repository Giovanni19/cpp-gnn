#pragma once
#include "VectorUtils.hpp"
#include "Graph.hpp"
#include "MLP.hpp"

#include <cstddef>

class MessagePassing
{
private:
    MLP edge_mlp;

public:
    MessagePassing(
        std::size_t node_dimension,
        std::size_t edge_dimension,
        std::size_t hidden_dimension
    );
    std::vector<std::vector<double>> update_edges(
        const Graph& graph
    ) const;
};