#include "Graph.hpp"
#include <algorithm>
void Graph::add_node(const Node& node)
{
    nodes.push_back(node);
}

void Graph::add_edge(const Edge& edge)
{
    edges.push_back(edge);
}

std::size_t Graph::num_nodes() const
{
    return nodes.size();
}

std::size_t Graph::num_edges() const
{
    return edges.size();
}

std::vector<int> Graph::neighbors(int node_id) const
{
    std::vector<int> result;

    // attraversa edges
    for (const Edge& edge : edges)
    {
        if (edge.source == node_id)
        {
            int candidate = edge.target;

            if (std::find(result.begin(), result.end(), candidate) == result.end())
            {
                result.push_back(candidate);
            }
        }

        if (edge.target == node_id)
        {
            int candidate = edge.source;

            if (std::find(result.begin(), result.end(), candidate) == result.end())
            {
                result.push_back(candidate);
            }
        }
    }

    return result;
}