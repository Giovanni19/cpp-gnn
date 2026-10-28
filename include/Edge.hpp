#pragma once

#include <vector>

struct Edge{
    int source;
    int target;
    std:: vector<double> features;
    Edge(int source0, int target0, std:: vector<double> features0): source(source0), target(target0), features(features0)
    {

    }
};