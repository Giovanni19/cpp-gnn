#include "MessagePassing.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "VectorUtils.hpp"
#include <memory>
#include <unordered_map>

namespace
{
// id del nodo -> posizione in graph.get_nodes()
std::unordered_map<int, std::size_t> node_index(const Graph& graph)
{
    std::unordered_map<int, std::size_t> index;
    const auto& nodes = graph.get_nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        index[nodes[i].id] = i;
    }
    return index;
}
}

MessagePassing::MessagePassing(
    std::size_t node_dimension,
    std::size_t edge_dimension,
    std::size_t hidden_dimension,
    unsigned seed
)
    : node_dimension(node_dimension), edge_dimension(edge_dimension)
{
    node_mlp.add_layer(std::make_unique<Linear>(node_dimension + edge_dimension, hidden_dimension, seed + 0));
    node_mlp.add_layer(std::make_unique<ReLU>());
    node_mlp.add_layer(std::make_unique<Linear>(hidden_dimension, node_dimension, seed + 1));

    std::size_t edge_input_dimension =
        2 * node_dimension + edge_dimension;

    edge_mlp.add_layer(
        std::make_unique<Linear>(
            edge_input_dimension,
            hidden_dimension,
            seed + 2
        )
    );

    edge_mlp.add_layer(
        std::make_unique<ReLU>()
    );

    edge_mlp.add_layer(
        std::make_unique<Linear>(
            hidden_dimension,
            edge_dimension,
            seed + 3
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

std::vector<std::vector<double>> MessagePassing::update_nodes(
    const Graph& graph,
    const std::vector<std::vector<double>>& edge_messages
) const
{
    const auto& nodes = graph.get_nodes();
    auto sums = aggregate(graph, edge_messages);

    std::vector<std::vector<double>> updated_nodes;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        updated_nodes.push_back(node_mlp.forward(concatenate(nodes[i].features, sums[i])));
    }
    return updated_nodes;
}

std::vector<std::vector<double>> MessagePassing::forward(const Graph& graph) const
{
    return update_nodes(graph, update_edges(graph));
}

std::vector<std::vector<double>> MessagePassing::aggregate(
    const Graph& graph,
    const std::vector<std::vector<double>>& edge_messages
) const
{
    auto index = node_index(graph);
    const auto& edges = graph.get_edges();
    std::vector<std::vector<double>> sums(graph.num_nodes(), std::vector<double>(edge_dimension, 0.0));

    for (std::size_t e = 0; e < edges.size(); ++e)
    {
        auto& sum = sums[index.at(edges[e].target)];
        for (std::size_t k = 0; k < edge_dimension; ++k)
        {
            sum[k] += edge_messages[e][k];
        }
    }
    return sums;
}

// I layer ricordano solo l'ultimo input, quindi prima di ogni backward
// rifacciamo il forward dello stesso nodo/arco.
std::vector<std::vector<double>> MessagePassing::backward(
    const Graph& graph,
    const std::vector<std::vector<double>>& grad_nodes
)
{
    const auto& nodes = graph.get_nodes();
    const auto& edges = graph.get_edges();
    auto index = node_index(graph);
    auto sums = aggregate(graph, update_edges(graph));

    std::vector<std::vector<double>> grad_inputs(nodes.size(), std::vector<double>(node_dimension, 0.0));
    std::vector<std::vector<double>> grad_sums(nodes.size());

    // node_mlp: input = [feature nodo, somma messaggi]
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        node_mlp.forward(concatenate(nodes[i].features, sums[i]));
        auto g = node_mlp.backward(grad_nodes[i]);

        for (std::size_t k = 0; k < node_dimension; ++k)
        {
            grad_inputs[i][k] += g[k];
        }
        grad_sums[i].assign(g.begin() + node_dimension, g.end());
    }

    // edge_mlp: la somma passa lo stesso gradiente a ogni messaggio entrante
    for (const Edge& edge : edges)
    {
        std::size_t s = index.at(edge.source);
        std::size_t t = index.at(edge.target);

        edge_mlp.forward(concatenate(nodes[s].features, nodes[t].features, edge.features));
        auto g = edge_mlp.backward(grad_sums[t]);

        for (std::size_t k = 0; k < node_dimension; ++k)
        {
            grad_inputs[s][k] += g[k];
            grad_inputs[t][k] += g[node_dimension + k];
        }
    }
    return grad_inputs;
}

void MessagePassing::step(double learning_rate)
{
    edge_mlp.step(learning_rate);
    node_mlp.step(learning_rate);
}
