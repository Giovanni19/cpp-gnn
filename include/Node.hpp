#pragma once

#include <vector>

struct Node{
    int id;
    std:: vector<double> features;
    Node(int id0, std:: vector<double> features0): id(id0), features(features0)
    {

    }
};