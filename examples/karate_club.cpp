#include "MessagePassing.hpp"
#include "Linear.hpp"
#include "Loss.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

// Classificazione dei nodi sul Karate Club di Zachary.
// Conosciamo solo la fazione dei due istruttori (nodi 0 e 33):
// la GNN deve dedurre quella di tutti gli altri dalla struttura del grafo.
int main(int argc, char** argv)
{
    const int num_nodes = 34;
    std::ifstream file(argc > 1 ? argv[1] : "data/karate_club.txt");
    if (!file)
    {
        std::cerr << "Impossibile aprire il file dei dati\n";
        return 1;
    }

    std::vector<int> labels(num_nodes);
    std::vector<std::pair<int, int>> edges;
    std::string line, kind;
    while (std::getline(file, line))
    {
        std::istringstream in(line);
        int a, b;
        if (!(in >> kind >> a >> b)) continue;   // salta commenti e righe vuote
        if (kind == "label") labels[a] = b;
        if (kind == "edge") edges.push_back({a, b});
    }

    // feature di un nodo: la sua riga della matrice di adiacenza (se stesso incluso).
    // Nodi della stessa comunita' hanno vicini in comune, quindi feature simili.
    std::vector<std::vector<double>> features(num_nodes, std::vector<double>(num_nodes, 0.0));
    for (int i = 0; i < num_nodes; ++i) features[i][i] = 1.0;
    for (auto [a, b] : edges) features[a][b] = features[b][a] = 1.0;

    Graph graph;
    for (int i = 0; i < num_nodes; ++i) graph.add_node(Node(i, features[i]));
    for (auto [a, b] : edges)
    {
        graph.add_edge(Edge(a, b, {}));
        graph.add_edge(Edge(b, a, {}));
    }

    const std::vector<int> train_nodes = {0, 33};

    // Un solo layer: le feature contengono gia' i vicini diretti, un secondo layer
    // aggiunge soprattutto parametri da memorizzare.
    // Media dei messaggi: con la somma i nodi con molti vicini (fino a 17) fanno esplodere i valori.
    MessagePassing layer(num_nodes, 0, 4, 16, 1, true);   // archi senza feature
    MLP readout;
    readout.add_layer(std::make_unique<Linear>(num_nodes, 2, 20));

    for (int epoch = 0; epoch <= 200; ++epoch)
    {
        auto output = layer.forward(graph);

        // loss solo sui nodi di training; gli altri ricevono gradiente zero
        std::vector<std::vector<double>> grad(num_nodes, std::vector<double>(num_nodes, 0.0));
        double loss = 0.0;
        for (int i : train_nodes)
        {
            std::vector<double> grad_logits;
            loss += softmax_cross_entropy(readout.forward(output[i]), labels[i], grad_logits);
            grad[i] = readout.backward(grad_logits);
        }

        layer.backward(graph, grad);
        layer.step(0.01);
        readout.step(0.01);

        if (epoch % 50 == 0)
        {
            std::cout << "epoch " << epoch << "  loss " << loss << "\n";
        }
    }

    // accuratezza sui 32 nodi mai visti in training
    auto output = layer.forward(graph);
    int correct = 0;
    for (int i = 1; i < num_nodes - 1; ++i)
    {
        auto logits = readout.forward(output[i]);
        int predicted = logits[1] > logits[0] ? 1 : 0;
        correct += predicted == labels[i];
    }
    double accuracy = correct / double(num_nodes - 2);
    std::cout << "accuratezza sui nodi non etichettati: " << correct << "/" << num_nodes - 2
              << " (" << accuracy * 100 << "%)\n";

    // Con questi seed 72%. Su 12 seed diversi: tra 72% e 94%, in media ~80%.
    assert(accuracy > 0.65);
    return 0;
}
