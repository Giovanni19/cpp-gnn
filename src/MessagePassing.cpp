#include "MessagePassing.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "VectorUtils.hpp"
#include <memory>

MessagePassing::MessagePassing(
    std::size_t node_dimension,
    std::size_t edge_dimension,
    std::size_t hidden_dimension
)
{
    std::size_t edge_input_dimension =
        2 * node_dimension + edge_dimension;

    edge_mlp.add_layer(
        std::make_unique<Linear>(
            edge_input_dimension,
            hidden_dimension
        )
    );

    edge_mlp.add_layer(
        std::make_unique<ReLU>()
    );

    edge_mlp.add_layer(
        std::make_unique<Linear>(
            hidden_dimension,
            edge_dimension
        )
    );
}

std::vector<std::vector<double>> MessagePassing::update_edges(
    const Graph& graph
) const
{
    std::vector<std::vector<double>> updated_edges;

    // ...
    
    return updated_edges;
}