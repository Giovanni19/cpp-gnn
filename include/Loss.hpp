#pragma once

#include <vector>

// Softmax + cross-entropy per una sola classe corretta.
// Restituisce la loss e scrive in grad dL/dlogits (= softmax - one_hot).
double softmax_cross_entropy(
    const std::vector<double>& logits,
    int label,
    std::vector<double>& grad
);
