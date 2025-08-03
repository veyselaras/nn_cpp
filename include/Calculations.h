#ifndef CALCULATIONS_H_
#define CALCULATIONS_H_

#include <vector>
#include <cmath>
#include "Node.h"
#include "NNParams.h"

class Calculations {
public:
    double reluFunc(double input);
    double reluDerivFunc(double input);
    void softMax(std::vector<Node>& outputLayer);
    void convolution(std::vector<Node>& inputLayer, std::vector<Node>& outputLayer);
};

#endif
