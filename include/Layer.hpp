#pragma once

#include <vector>

class Layer
{
public:
    virtual std::vector<double> forward(
        const std::vector<double>& input
    ) const = 0;

    // Riceve dL/doutput (dell'ultimo forward), accumula i gradienti dei parametri
    // e restituisce dL/dinput.
    virtual std::vector<double> backward(
        const std::vector<double>& grad_output
    ) = 0;

    // Aggiorna i parametri (SGD) e azzera i gradienti accumulati.
    virtual void step(double /*learning_rate*/) {}

    virtual ~Layer() = default;
};