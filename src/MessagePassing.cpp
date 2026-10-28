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

    for (const Edge& edge : graph.get_edges())
    {
        const Node& source = graph.get_node(edge.source);
        const Node& target = graph.get_node(edge.target);

        std::vector<double> edge_input = concatenate(
            source.features,
            target.features,
            edge.features
        );
        std::vector<double> updated_features = edge_mlp.forward(edge_input);
        updated_edges.push_back(updated_features);
    }
    
    return updated_edges;
}