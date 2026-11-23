#include "Loss.hpp"

#include <algorithm>
#include <cmath>

double softmax_cross_entropy(
    const std::vector<double>& logits,
    int label,
    std::vector<double>& grad
)
{
    // sottraggo il massimo per stabilita' numerica
    double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;

    grad.resize(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i)
    {
        grad[i] = std::exp(logits[i] - max_logit);
        sum += grad[i];
    }
    for (double& p : grad)
    {
        p /= sum;
    }

    double loss = -std::log(grad[label]);
    grad[label] -= 1.0;
    return loss;
}
