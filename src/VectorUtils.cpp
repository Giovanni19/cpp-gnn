#include "VectorUtils.hpp"

std::vector<double> concatenate(
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
   std::vector<double> result = a;
   result.insert(result.end(), b.begin(), b.end());
   return result;
}

std::vector<double> concatenate(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c
)
{
    std::vector<double> result;

    result.reserve(
        a.size() + b.size() + c.size()
    );

    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    result.insert(result.end(), c.begin(), c.end());

    return result;
}