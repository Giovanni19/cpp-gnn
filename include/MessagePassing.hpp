#pragma once

#include "Graph.hpp"
#include "MLP.hpp"

#include <cstddef>
#include <random>

class MessagePassing
{
private:
    MLP edge_mlp;
    MLP node_mlp;
    std::size_t node_dimension;
    std::size_t edge_dimension;

    // Somma dei messaggi entranti per ogni nodo (stesso ordine di graph.get_nodes()).
    std::vector<std::vector<double>> aggregate(
        const Graph& graph,
        const std::vector<std::vector<double>>& edge_messages
    ) const;

public:
    MessagePassing(
        std::size_t node_dimension,
        std::size_t edge_dimension,
        std::size_t hidden_dimension,
        unsigned seed = std::random_device{}()
    );
    std::vector<std::vector<double>> update_edges(
        const Graph& graph
    ) const;

    // Somma i messaggi degli archi entranti in ogni nodo e aggiorna le feature
    // (stesso ordine di graph.get_nodes()).
    std::vector<std::vector<double>> update_nodes(
        const Graph& graph,
        const std::vector<std::vector<double>>& edge_messages
    ) const;

    // update_edges + update_nodes.
    std::vector<std::vector<double>> forward(const Graph& graph) const;

    // Riceve dL/d(nuove feature) per ogni nodo, accumula i gradienti dei due MLP
    // e restituisce dL/d(feature di input) per ogni nodo.
    std::vector<std::vector<double>> backward(
        const Graph& graph,
        const std::vector<std::vector<double>>& grad_nodes
    );

    void step(double learning_rate);
};